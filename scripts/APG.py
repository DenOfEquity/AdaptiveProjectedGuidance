import torch
import gradio as gr

from modules import scripts, shared
from backend.sampling.condition import Condition, compile_conditions
from backend.sampling.sampling_function import calc_cond_uncond_batch
from modules.prompt_parser import SdConditioning
from modules.ui_components import InputAccordion


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
            apg_method = gr.Radio(label='CFG method', choices=["APG", "TraSCE", "Normal"], value="APG")

            apg_eta = gr.Slider(label='eta (contrast)', minimum=-1.0, maximum=1, step=0.01, value=0.0)
            apg_r   = gr.Slider(label='rescale threshold', minimum=0, maximum=20, step=0.01, value=8.0)
            apg_m   = gr.Slider(label='momentum', minimum=-1.0, maximum=1.0, step=0.01, value=-0.5)
            with gr.Row():
                apg_icg = gr.Slider(label='image Independent Condition Guidance', minimum=0.0, maximum=0.2, step=0.001, value=0.0)
                apg_icg_s = gr.Slider(label='ICG start', minimum=0.0, maximum=1.0, step=0.01, value=0.4)
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
            visible = True if method == "APG" else False
            return gr.update(visible=visible), gr.update(visible=visible), gr.update(visible=visible)

        apg_method.change(fn=show_options, inputs=[apg_method], outputs=[apg_eta, apg_r, apg_m], show_progress=False)

        apg_enabled.do_not_save_to_config = True
        apg_method.do_not_save_to_config = True
        apg_eta.do_not_save_to_config = True
        apg_r.do_not_save_to_config = True
        apg_m.do_not_save_to_config = True
        apg_icg.do_not_save_to_config = True
        apg_icg_s.do_not_save_to_config = True

        self.infotext_fields = [
            (apg_enabled, lambda d: d.get("APG_enabled", False)),
            (apg_method,   "APG_method"),
            (apg_eta,      "APG_eta"),
            (apg_r,        "APG_r"),
            (apg_m,        "APG_m"),
            (apg_icg,      "APG_ICG"),
            (apg_icg_s,    "APG_ICG_start"),
        ]

        return apg_enabled, apg_method, apg_eta, apg_r, apg_m, apg_icg, apg_icg_s
        
    def process(self, p, *script_args, **kwargs):
        apg_enabled, apg_method, apg_eta, apg_r, apg_m, apg_icg, apg_icg_s = script_args

        if apg_enabled:
            p.extra_generation_params.update(dict(
                APG_enabled   = apg_enabled,
                APG_method    = apg_method,
                APG_ICG       = apg_icg,
                APG_ICG_start = apg_icg_s,
            ))
            if apg_method == "APG":
                p.extra_generation_params.update(dict(
                    APG_eta       = apg_eta,
                    APG_r         = apg_r,
                    APG_m         = apg_m,
                ))

        return

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        apg_enabled, apg_method, apg_eta, apg_r, apg_m, apg_icg, apg_icg_s = script_args

        if not apg_enabled:
            return

        def patch(model, eta, r, m, icg, icg_start):
            apg = APG(eta, r, m)
            start = model.model.predictor.percent_to_sigma(icg_start)
            
            def sampler_apg(args):
                input = args["input"]
                cond = args["cond_denoised"]
                uncond = args["uncond_denoised"]
                cond_scale = args["cond_scale"]
                sigma = args["sigma"]
                options = args["model_options"]
                
                if icg > 0 and sigma <= start:
                    factor = icg
                    if apg_method != "APG":
                        factor *= 0.1
                    factor *= min(1.0, sigma)

                    icg_uncond = torch.randn_like(uncond)
                    icg_uncond *= uncond.std()
                    torch.lerp(uncond, icg_uncond, factor, out=uncond)

                match apg_method:
                    case "APG":
                        denoised =  apg.normalized_guidance(cond, uncond, cond_scale, apg.momentum, apg.eta, apg.r)
                    case "TraSCE":
                        bias, _ = calc_cond_uncond_batch(model.model, APGforForge.empty, None, input, sigma, options)
                        denoised = bias + cond_scale * (cond - uncond)
                    case _:
                        denoised = uncond + cond_scale * (cond - uncond)

                return input - denoised

            m = model.clone()
            m.set_model_sampler_cfg_function(sampler_apg)
            return (m, )


        unet = p.sd_model.forge_objects.unet
        unet = patch(unet, apg_eta, apg_r, apg_m, apg_icg, apg_icg_s)[0]
        p.sd_model.forge_objects.unet = unet

        empty_prompt = SdConditioning([""], is_negative_prompt=False, width=p.width, height=p.height)
        empty_cond = shared.sd_model.get_learned_conditioning(empty_prompt)
        APGforForge.empty = compile_conditions(empty_cond)

        return

