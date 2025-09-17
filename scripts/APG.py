import torch, math
import gradio as gr
import copy

from modules import scripts, shared
from backend.sampling.condition import Condition, compile_conditions
import backend.sampling.sampling_function
from backend.sampling.sampling_function import calc_cond_uncond_batch
from modules.prompt_parser import SdConditioning
from modules.ui_components import InputAccordion
from modules.script_callbacks import on_cfg_denoiser, remove_current_script_callbacks


#### Frequency-Decoupled Guidance: https://arxiv.org/pdf/2506.19713
####  Seyedmorteza Sadat, Tobias Vontobel, Farnood Salehi, Romann M. Weber

from kornia.geometry import pyrup
try:
    from kornia.geometry.transform.pyramid import build_laplacian_pyramid
    apg_functions = ["APG", "TraSCE", "method two", "FDG", "Normal"]
except:
    apg_functions = ["APG", "TraSCE", "method two", "Normal"]
   

def project(
    v0: torch.Tensor, # [B, C,H, W]
    v1: torch.Tensor, # [B, C,H, W]
):
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1,-2,-3])
    v0_parallel = (v0 * v1).sum(dim=[-1,-2,-3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def build_image_from_pyramid(pyramid):
    img = pyramid[-1]
    for i in range(len(pyramid)-2,-1,-1):
        img = pyrup(img) + pyramid[i]
    return img

# We assume all model predictions are converted to "x_0" prediction.
def laplacian_guidance(
    pred_cond: torch.Tensor, # [B, C, H, W]
    pred_uncond:torch.Tensor, # [B, C, H, W]
    guidance_scale=[1.0, 1.0], # Guidance scales from high- to low-frequency
    # parallel_weights=None, # Optionalweights for projection
):
    levels = len(guidance_scale)
    # if parallel_weights = None:
        # parallel_weights= [1.0] * levels

    pred_cond_pyramid = build_laplacian_pyramid(pred_cond, levels)
    pred_uncond_pyramid = build_laplacian_pyramid(pred_uncond, levels)

    pred_guided_pyramid = []
    parameters = zip(pred_cond_pyramid,pred_uncond_pyramid, guidance_scale)#, parallel_weights)
    for idx, (p_cond, p_uncond, scale) in enumerate(parameters):
        diff = p_cond - p_uncond
        diff_parallel, diff_orthogonal = project(diff, p_cond)
        diff = diff_parallel + diff_orthogonal
        p_guided = p_cond + (scale-1) * diff
        pred_guided_pyramid.append(p_guided)
    pred_guided = build_image_from_pyramid(pred_guided_pyramid)

    pred_guided = pred_guided[:, :, :pred_cond.shape[2], :pred_cond.shape[3]]

    return pred_guided.to(pred_cond.dtype)

#### end FDG


class MomentumBuffer:
    def __init__(self, momentum: float):
        self.momentum = momentum
        self.running_average = 0

    def update(self, update_value: torch.Tensor):
        new_average = self.momentum * self.running_average
        self.running_average = update_value + new_average

class APG:
    def __init__(self, eta, r, m):
        self.eta = eta
        self.r = r
        self.momentum = MomentumBuffer(m)

    def project(
        self,
        v0: torch.Tensor, # [B, C, H, W]
        v1: torch.Tensor, # [B, C, H, W]
    ):
        dtype = v0.dtype
        v0, v1 = v0.double(), v1.double()
        v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
        v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
        v0_orthogonal = v0 - v0_parallel
        return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

    def normalized_guidance(
        self,
        pred_cond: torch.Tensor, # [B, C, H, W]
        pred_uncond: torch.Tensor, # [B, C, H, W]
        guidance_scale: float,
        momentum_buffer: MomentumBuffer = None,
        eta: float = 1.0,
        norm_threshold: float = 0.0,
    ):
        diff = pred_cond - pred_uncond
        if momentum_buffer is not None:
            try:
                momentum_buffer.update(diff)
            except:
                pass
            diff = momentum_buffer.running_average
        if norm_threshold > 0:
            ones = torch.ones_like(diff)
            diff_norm = diff.norm(p=2, dim=[-1, -2, -3], keepdim=True)
            scale_factor = torch.minimum(ones, norm_threshold / diff_norm)
            diff = diff * scale_factor
            try:
                diff_parallel, diff_orthogonal = self.project(diff, pred_cond)
                normalized_update = diff_orthogonal + eta * diff_parallel
                pred_guided = pred_cond + (guidance_scale - 1) * normalized_update
            except:
                pred_guided = pred_cond
        else:
            pred_guided = pred_cond

        return pred_guided

class APGforForge(scripts.Script):
    sorting_priority = 11.9
    empty = None
    storeCFG = 1.0
    CFGweight = 1.0
    backup_sampling_function_inner = None

    def __init__(self):
        if APGforForge.backup_sampling_function_inner == None:
            APGforForge.backup_sampling_function_inner = backend.sampling.sampling_function.sampling_function_inner

    presets_builtin = [
        #   name, eta, rescale threshold, momentum
        ('SD 1.5', 0.0, 6.5, -0.5),
        ('SD 1.5 CFG++', 0.0, 2.5, -0.45),
        ('SD 2.1', 0.0, 7.5, -0.75),
        ('SDXL',   0.0, 15,  -0.5),
    ]
    try:
        import apg_presets
        presets = presets_builtin + apg_presets.presets_custom
    except:
        presets = presets_builtin

    def title(self):
        return "Adaptive Projected Guidance"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):

        with InputAccordion(False, label=self.title()) as apg_enabled:
            apg_method = gr.Radio(label='CFG method', choices=apg_functions, value="APG")

            with InputAccordion(False, label='CFG fade') as fade_enabled:
                with gr.Row():
                    lowCFG1   = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.1, label='CFG 1 until step')
                    maxScale  = gr.Slider(minimum=1.0, maximum=4.0,  step=0.01, value=1.0, label='boost factor')
                with gr.Row():
                    boostStep = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.2, label='CFG boost start step')
                    minScale  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=1.0, label='fade factor')
                with gr.Row():
                    highStep  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.4, label='full boost at step')
                    heuristic = gr.Slider(minimum=0.0, maximum=16.0, step=0.1,  value=0,   label='Heuristic CFG (for Normal)')
                with gr.Row():
                    fadeStep  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.5, label='CFG fade start step')
                    hStart    = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.0, label='... start step for heuristic CFG')
                with gr.Row():
                    zeroStep  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.7, label='full fade at step')
                    reinhard  = gr.Slider(minimum=0.0, maximum=16.0, step=0.1,  value=0.0, label='Reinhard CFG (for Normal)')
                with gr.Row():
                    highCFG1  = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.8, label='CFG 1 after step')
                    rescale   = gr.Slider(minimum=0.0, maximum=1.0,  step=0.01, value=0.0, label='Rescale CFG (for Normal)')

                with gr.Row():
                    cntrMean   = gr.Checkbox(value=False, label='centre conds to mean') #add other methods?

            apg_eta = gr.Slider(label='eta (contrast)', minimum=-1.0, maximum=1, step=0.01, value=0.0)
            apg_r   = gr.Slider(label='rescale threshold', minimum=0, maximum=20, step=0.01, value=8.0)
            apg_m   = gr.Slider(label='momentum', minimum=-1.0, maximum=1.0, step=0.01, value=-0.5)

            with gr.Row(visible=False) as fdg:
                apg_fdg_scale = gr.Textbox(label='Frequency-Decoupled Guidance scaler (high to low; replaces CFG scale)', value="*1.1, 1.5", interactive=True, max_lines=1)

            with gr.Row():
                apg_icg = gr.Slider(label='image Independent Condition Guidance', minimum=0.0, maximum=0.2, step=0.001, value=0.0)
                apg_icg_s = gr.Slider(label='ICG start', minimum=0.0, maximum=1.0, step=0.01, value=0.4)
            with gr.Row():
                apg_star = gr.Checkbox(label='CFG *', value=False)
                apg_tdamp = gr.Checkbox(label='Tangential Damping', value=False)

            apg_post_cfg = gr.Radio(label='post CFG method', choices=["MaHiRo", "SLG (SD3)", "None"], value="None")
            with gr.Row(visible=False) as slg_1:
                apg_slg_scale = gr.Slider(label='Skip Layer Guidance', minimum=0.0, maximum=16.0, step=0.1, value=2.7)
                apg_slg_layers = gr.Textbox(label='SLG layers', value="7, 8, 9", interactive=True, max_lines=1)
            with gr.Row(visible=False) as slg_2:
                apg_slg_start = gr.Slider(label='SLG start', minimum=0.0, maximum=1.0, step=0.01, value=0.01)
                apg_slg_end = gr.Slider(label='SLG end', minimum=0.0, maximum=1.0, step=0.01, value=0.2)
            with gr.Row():
                apg_preset = gr.Dropdown(label='', choices=[x[0] for x in APGforForge.presets], value='(APG presets)', type='index', scale=0, allow_custom_value=True)

        def setParams (preset):
            if preset < len(APGforForge.presets):
                return  'APG', APGforForge.presets[preset][1], APGforForge.presets[preset][2], APGforForge.presets[preset][3], \
                        '(APG presets)'
            else:
                return 'APG', 0.0, 8.0, -0.5, '(APG presets)'

        apg_preset.input(fn=setParams, inputs=[apg_preset],
                         outputs=[apg_method, apg_eta, apg_r, apg_m, apg_preset], show_progress=False)

        def show_options (method):
            APGvisible = True if "APG" in method else False
            FDGvisible = True if "FDG" in method else False
            return gr.update(visible=APGvisible), gr.update(visible=APGvisible), gr.update(visible=APGvisible), gr.update(visible=FDGvisible)

        def show_post_cfg (method):
            visible = True if method == "SLG (SD3)" else False
            return gr.update(visible=visible), gr.update(visible=visible)

        apg_method.change(fn=show_options, inputs=[apg_method], outputs=[apg_eta, apg_r, apg_m, fdg], show_progress=False)
        apg_post_cfg.change(fn=show_post_cfg, inputs=[apg_post_cfg], outputs=[slg_1, slg_2], show_progress=False)

        self.infotext_fields = [
            (apg_enabled, lambda d: d.get("APG_enabled", False)),
            (apg_method,    "APG_method"),
            (apg_post_cfg,  "APG_post_cfg"),
            (apg_eta,       "APG_eta"),
            (apg_r,         "APG_r"),
            (apg_m,         "APG_m"),
            (apg_fdg_scale, "APG_FDG_scale"),
            (apg_icg,       "APG_ICG"),
            (apg_icg_s,     "APG_ICG_start"),
            (apg_star,      "APG_CFGstar"),
            (apg_tdamp,     "APG_Tangential_Damping"),
            (apg_slg_scale, "APG_SLG_scale"),
            (apg_slg_layers,"APG_SLG_layers"),
            (apg_slg_start, "APG_SLG_start"),
            (apg_slg_end,   "APG_SLG_end"),

            (fade_enabled, lambda d: d.get("apg_fade_enabled", False)),
            (cntrMean,  "apg_fade_cntrMean"),
            (boostStep, "apg_fade_boostStep"),
            (highStep,  "apg_fade_highStep"),
            (maxScale,  "apg_fade_maxScale"),
            (fadeStep,  "apg_fade_fadeStep"),
            (zeroStep,  "apg_fade_zeroStep"),
            (minScale,  "apg_fade_minScale"),
            (lowCFG1,   "apg_fade_lowCFG1"),
            (highCFG1,  "apg_fade_highCFG1"),
            (reinhard,  "apg_fade_reinhard"),
            (rescale,   "apg_fade_rescale"),
            (heuristic, "apg_fade_heuristic"),
            (hStart,    "apg_fade_hStart"),
        ]

        return (apg_enabled, apg_method, apg_post_cfg, apg_eta, apg_r, apg_m, apg_fdg_scale, apg_icg, apg_icg_s, apg_star, apg_tdamp, 
                apg_slg_scale, apg_slg_layers, apg_slg_start, apg_slg_end,
                fade_enabled, cntrMean, boostStep, highStep, maxScale, fadeStep, zeroStep, minScale, lowCFG1, highCFG1, reinhard, rescale, heuristic, hStart)

    def denoiser_callback(self, params):
        lastStep = params.total_sampling_steps - 1
        thisStep = params.sampling_step
        sigma = params.sigma[0]
        
        lowCFG1   = self.lowCFG1   * lastStep
        highStep  = self.highStep  * lastStep
        boostStep = self.boostStep * lastStep
        highCFG1  = self.highCFG1  * lastStep
        fadeStep  = self.fadeStep  * lastStep
        zeroStep  = self.zeroStep  * lastStep

        if thisStep < lowCFG1:
            boostWeight = 0.0
        elif thisStep < boostStep:
            boostWeight = 1.0
        elif thisStep < highStep:
            boostWeight = 1.0 + (self.maxScale - 1.0) * ((thisStep - boostStep) / (highStep - boostStep))
        else:
            boostWeight = self.maxScale

        if thisStep > highCFG1:
            fadeWeight = 0.0
        else:
            if thisStep < fadeStep:
                fadeWeight = 1.0
            elif thisStep < zeroStep:
                fadeWeight = 1.0 - (thisStep - fadeStep) / (zeroStep  - fadeStep)
            else:
                fadeWeight = 0.0

            # at this point, weight is in the range 0.0->1.0
            fadeWeight *= (1.0 - self.minScale)
            fadeWeight += self.minScale
            # now it is minimum->1.0

        APGforForge.CFGweight = boostWeight * fadeWeight

    def process(self, p, *script_args, **kwargs):
        (apg_enabled, apg_method, apg_post_cfg, apg_eta, apg_r, apg_m, apg_fdg_scale, apg_icg, apg_icg_s, apg_star, tangential_damp, 
        apg_slg_scale, apg_slg_layers, apg_slg_start, apg_slg_end,
        fade_enabled, cntrMean, boostStep, highStep, maxScale, fadeStep, zeroStep, minScale, lowCFG1, highCFG1, reinhard, rescale, heuristic, hStart) = script_args

        if apg_enabled:
            p.extra_generation_params.update(dict(
                APG_enabled   = apg_enabled,
                APG_method    = apg_method,
                APG_post_cfg  = apg_post_cfg,
                APG_CFGstar   = apg_star,
                APG_Tangential_Damping = tangential_damp,
            ))
            if apg_icg > 0.0:
                p.extra_generation_params.update(dict(
                    APG_ICG       = apg_icg,
                    APG_ICG_start = apg_icg_s,
                ))
            if "APG" in apg_method:
                p.extra_generation_params.update(dict(
                    APG_eta       = apg_eta,
                    APG_r         = apg_r,
                    APG_m         = apg_m,
                ))
            if "FDG" in apg_method:
                p.extra_generation_params.update(dict(
                    APG_FDG_scale = apg_fdg_scale,
                ))
            if apg_post_cfg == "SLG (SD3)":
                p.extra_generation_params.update(dict(
                    APG_SLG_scale   = apg_slg_scale,
                    APG_SLG_layers  = apg_slg_layers,
                    APG_SLG_start   = apg_slg_start,
                    APG_SLG_end     = apg_slg_end,
                ))

            self.boostStep  = boostStep
            self.highStep   = highStep
            self.maxScale   = maxScale
            self.fadeStep   = fadeStep
            self.zeroStep   = zeroStep
            self.minScale   = minScale
            self.lowCFG1    = lowCFG1
            self.highCFG1   = highCFG1
            APGforForge.heuristic  = heuristic
            APGforForge.reinhard   = reinhard

            # logs, could save boost start/full only if boost factor > 1
            #       similar for fade
            if fade_enabled:
                p.extra_generation_params.update(dict(
                    apg_fade_enabled   = fade_enabled,
                    apg_fade_cntrMean  = cntrMean,
                    apg_fade_boostStep = boostStep,
                    apg_fade_highStep  = highStep,
                    apg_fade_maxScale  = maxScale,
                    apg_fade_fadeStep  = fadeStep,
                    apg_fade_zeroStep  = zeroStep,
                    apg_fade_minScale  = minScale,
                    apg_fade_lowCFG1   = lowCFG1,
                    apg_fade_highCFG1  = highCFG1,
                    apg_fade_reinhard  = reinhard,
                    apg_fade_rescale   = rescale,
                    apg_fade_heuristic = heuristic,
                    apg_fade_hStart    = hStart,
                ))
                #   must log the parameters before fixing minScale
                self.minScale /= self.maxScale

                on_cfg_denoiser(self.denoiser_callback)

                backend.sampling.sampling_function.sampling_function_inner = APGforForge.sampling_function_inner

        return


#   edited from backend/sampling/sampling_function.py to add cond_scaling (initial 3 lines)
    def sampling_function_inner(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, return_full=False):
        cond_scale *= APGforForge.CFGweight
        if cond_scale < 1.0:
            cond_scale = 1.0

        edit_strength = sum((item['strength'] if 'strength' in item else 1) for item in cond)

        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        else:
            uncond_ = uncond

        for fn in model_options.get("sampler_pre_cfg_function", []):
            model, cond, uncond_, x, timestep, model_options = fn(model, cond, uncond_, x, timestep, model_options)

        cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)

        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        elif not math.isclose(edit_strength, 1.0):
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
        else:
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        if return_full:
            return cfg_result, cond_pred, uncond_pred

        return cfg_result


    def process_before_every_sampling(self, p, *script_args, **kwargs):
        (apg_enabled, apg_method, apg_post_cfg, apg_eta, apg_r, apg_m, apg_fdg_scale, apg_icg, apg_icg_s, apg_star, tangential_damp,
        apg_slg_scale, apg_slg_layers, apg_slg_start, apg_slg_end,
        fade_enabled, cntrMean, boostStep, highStep, maxScale, fadeStep, zeroStep, minScale, lowCFG1, highCFG1, reinhard, rescale, heuristic, hStart) = script_args

        if not apg_enabled:
            return

        def patch(model, eta, r, m, icg, icg_start, cfg_star, tangential_damp):
            apg = APG(eta, r, m)

            def sampler_apg(args):
                input = args["input"]
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                cond_scale = args["cond_scale"]
                sigma = args["sigma"]
                options = args["model_options"]
                
                APGforForge.storeCFG = cond_scale   # for MaHiRo post_cfg

                if cntrMean == True:
                    for b in range(len(cond)):
                        for c in range(4):
                            cond[b][c] -= cond[b][c].mean()
                            uncond[b][c] -= uncond[b][c].mean()
                
                if cond_scale > 1.0:
                    if icg > 0 and shared.state.sampling_step / (shared.state.sampling_steps - 1) >= icg_start:
                        factor = icg
                        if apg_method != "APG":
                            factor *= 0.1
                        factor *= min(1.0, sigma)

                        icg_uncond = torch.randn_like(uncond)
                        icg_uncond *= uncond.std()
                        torch.lerp(uncond, icg_uncond, factor, out=uncond)

                    if cfg_star:
                        batch_size = cond.shape[0]
                        cond_flat = cond.view(batch_size, -1)  
                        uncond_flat = uncond.view(batch_size, -1)  
                        # Calculate dot product
                        dot_product = torch.sum(cond_flat * uncond_flat, dim=1, keepdim=True)

                        # Squared norm of uncondition
                        squared_norm = torch.sum(uncond_flat ** 2, dim=1, keepdim=True) + 1e-8

                        st_star = dot_product / squared_norm

                        uncond *= st_star.view(batch_size, 1, 1, 1)

                    if tangential_damp:
                        #https://arxiv.org/pdf/2503.18137
                        # Mingi Kwon, Shin seong Kim, Yi Ting Hsiao, Jaeseok Jeong, Youngjung Uh
 
                        all_noise = torch.stack((cond, uncond),dim=1).to(dtype=torch.float32)
                        all_noise = all_noise.reshape(all_noise.size(0), all_noise.size(1), -1)

                        U, S, Vh = torch.linalg.svd(all_noise, full_matrices=False)
                        Vh = Vh.to(all_noise.device)
                        Vh_modified = Vh.clone().to(all_noise.device)
                        Vh_modified[:,1] = 0

                        noise_null_flat = uncond.reshape(uncond.size(0), 1, -1).to(dtype=torch.float32)
                        noise_null_flat = noise_null_flat.to(Vh.device)

                        x_Vh = torch.matmul(noise_null_flat, Vh.transpose(-2,-1))
                        x_Vh_V = torch.matmul(x_Vh, Vh_modified)

                        uncond = x_Vh_V.reshape(*uncond.shape).to(cond.device, dtype=cond.dtype)
 
                    args["uncond"] = input - uncond

                match apg_method:
                    case "APG":
                        denoised = apg.normalized_guidance(cond, uncond, cond_scale, apg.momentum, apg.eta, apg.r)
                        return input - denoised
                    case "TraSCE":
                        if cond_scale > 1.0:
                            bias, _ = calc_cond_uncond_batch(model.model, APGforForge.empty, None, input, sigma, options)
                            denoised = bias + cond_scale * (cond - uncond)
                        else:
                            denoised = cond
                        return input - denoised
                    case "method two":
                        if cond_scale > 1.0:
                            bias, _ = calc_cond_uncond_batch(model.model, APGforForge.empty, None, input, sigma, options)
                            cond_scale *= 0.5
                            denoised = (2*cond_scale + 1.0) * cond - cond_scale * (bias + uncond)
                        else:
                            denoised = cond
                        return input - denoised
                    case "FDG":
                        FDG_scale = []
                        for num in apg_fdg_scale.split(','):
                            if num.startswith('*'):
                                FDG_scale.append(cond_scale * float(num[1:]))
                            else:
                                FDG_scale.append(float(num))

                        if len(FDG_scale) <= 1:
                            FDG_scale = [cond_scale, cond_scale]
                        denoised = laplacian_guidance(cond, uncond, FDG_scale)
                        return input - denoised
                    case _:
                        # denoised = uncond + cond_scale * (cond - uncond)
                        if cond_scale > 1.0:
                            nonlocal hStart, rescale
                            cond = args["cond"]
                            uncond = args["uncond"]
                            heuristic = APGforForge.heuristic * APGforForge.CFGweight
                            reinhard = APGforForge.reinhard * APGforForge.CFGweight

            #   cond_scale weighting now applied in sampling_function_inner, can avoid processing of uncond for performance increase

                            thisStep = shared.state.sampling_step
                            lastStep = shared.state.sampling_steps - 1
                            
                            noisePrediction = cond - uncond
                            
            #   heuristic scaling, higher hcfg acts to boost contrast/detail/sharpness; low reduces; quantile has effect, but not significant for quality IMO
                            if heuristic != 0.0 and heuristic != cond_scale and thisStep >= hStart * lastStep:
                                base = uncond + cond_scale * noisePrediction
                                heur = uncond + heuristic * noisePrediction

                                #   center both on zero
                                # if cntrMean:
                                    # base = base - base.mean()
                                    # heur = heur - heur.mean()

                                #   calc 99.0% quartiles - doesn't seem to have value as an option
                                baseQ = torch.quantile(base.abs(), 0.99)
                                heurQ = torch.quantile(heur.abs(), 0.99)
                                del base, heur

                                if baseQ != 0.0 and heurQ != 0.0:
                                    noisePrediction *= baseQ / heurQ
                                del baseQ, heurQ
            #   end: heuristic scaling

            #   reinhard tonemap from comfy
                            if reinhard != 0.0 and reinhard != cond_scale:
                                multiplier = 1.0 / cond_scale * reinhard
                                noise_pred_vector_magnitude = (torch.linalg.vector_norm(noisePrediction, dim=(1)) + 0.0000000001)[:,None]
                                noisePrediction /= noise_pred_vector_magnitude

                                mean = torch.mean(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                                std = torch.std(noise_pred_vector_magnitude, dim=(1,2,3), keepdim=True)
                                top = (std * 3 + mean) * multiplier

                                noise_pred_vector_magnitude *= (1.0 / top)
                                new_magnitude = noise_pred_vector_magnitude / (noise_pred_vector_magnitude + 1.0)
                                new_magnitude *= top
                                cond_scale *= new_magnitude
            #   end: reinhard

            #   rescaleCFG
                            denoised = uncond + cond_scale * noisePrediction
                            if rescale != 0.0:
                                ro_pos = torch.std(cond, dim=(1,2,3), keepdim=True)
                                ro_cfg = torch.std(denoised, dim=(1,2,3), keepdim=True)

                                if ro_pos != 0.0 and ro_cfg != 0.0:
                                    x_rescaled = denoised * (ro_pos / ro_cfg)
                                    denoised = torch.lerp (denoised, x_rescaled, rescale)
                                    del x_rescaled

                                del ro_pos, ro_cfg
            #   end: rescaleCFG
                            del noisePrediction

                            return denoised

            #   end: if cond_scale > 1.0
                        else:
                            return input - cond

            def post_cfg_apg(args):
                denoised, cond, cond_denoised, sigma, x, options = \
                    args["denoised"], args["cond"], args["cond_denoised"], args["sigma"], args["input"], args["model_options"]

                match apg_post_cfg:
                    case "MaHiRo":
                        #   via ForgeClassic, via ComfyUI, original by yoinked-h
                        scale = APGforForge.storeCFG    # should this be independant?
                        uncond_denoised: torch.Tensor = args["uncond_denoised"]
                        leap = cond_denoised * scale
                        u_leap = uncond_denoised * scale

                        merge = (leap + denoised) / 2
                        normu = torch.sqrt(u_leap.abs()) * u_leap.sign()
                        normm = torch.sqrt(merge.abs()) * merge.sign()
                        sim = torch.nn.functional.cosine_similarity(normu, normm).mean()
                        simsc = 2 * (sim + 1)
                        denoised = (simsc * denoised + (4 - simsc) * leap) / 4

                    case "SLG (SD3)":
                        if sigma <= model.model.predictor.percent_to_sigma(apg_slg_start) and sigma >= model.model.predictor.percent_to_sigma(apg_slg_end):
                            slg_options = copy.deepcopy(options)
                            slg_options["transformer_options"]["skip_layers"] = [int(num) for num in apg_slg_layers.split(',')]
                            SLG, _ = calc_cond_uncond_batch(model.model, cond, None, x, sigma, slg_options)

                            denoised = denoised + (cond_denoised - SLG) * apg_slg_scale
# add PPAG?
                return denoised

            m = model.clone()
            m.set_model_sampler_cfg_function(sampler_apg)
            if apg_post_cfg != "None":
                m.set_model_sampler_post_cfg_function(post_cfg_apg)
            return (m, )


        unet = p.sd_model.forge_objects.unet
        unet = patch(unet, apg_eta, apg_r, apg_m, apg_icg, apg_icg_s, apg_star, tangential_damp)[0]
        p.sd_model.forge_objects.unet = unet

        if apg_method == "TraSCE" or apg_method == "method two":
            empty_prompt = SdConditioning([""], is_negative_prompt=False, width=p.width, height=p.height)
            empty_cond = shared.sd_model.get_learned_conditioning(empty_prompt)
            APGforForge.empty = compile_conditions(empty_cond)

        return


    def postprocess(self, params, processed, *args):
        enabled = args[0]
        if enabled: # strictly: if fade_enabled, but no harm in always tidying
            if APGforForge.backup_sampling_function_inner != None:
                backend.sampling.sampling_function.sampling_function_inner = APGforForge.backup_sampling_function_inner

        remove_current_script_callbacks()
        return
