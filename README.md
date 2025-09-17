## Adaptive Projected Guidance for Forge2 webUI ##
### APG code from ELIMINATING OVERSATURATION AND ARTIFACTS OF HIGH GUIDANCE SCALES IN DIFFUSION MODELS ###
### credit to Seyedmorteza Sadat, Otmar Hilliges, Romann M. Weber ###
### https://arxiv.org/pdf/2410.02416 ###
---
controls described in the paper

eta affects saturation/contrast

rescale threshold ... highly variable depending on model, experiment

momentum between -0.75 and -0.25 recommended

---
The paper mostly compares APG with too-high CFG, which I'm not sure is all that useful - we already know high CFG burns badly, so a better comparison would be between APG and reasonable CFG to see if we really do get better quality, better prompt adherence, etc.
Anyway, it does *something*.

#### sdxl model haveall cfg3.3 ####
![](lavender2.png) 

#### sd1.5 model swizz8real cfg 15 ####
![](woodelves2.png) 


---
Also, 

### TraSCE: Trajectory Steering for Concept Erasure https://arxiv.org/abs/2412.07658 ###

Intent from the paper is to improve the ability of the negative prompt to remove/control concepts: using a negative is necessary - no effect if the negative prompt is empty.

Basically, `cfg_result = empty_conditioning + guidance * (positive_conditioning - negative_conditioning)`.

Standard is `cfg_result = negative_conditioning + guidance * (positive_conditioning - negative_conditioning)`.

---
Also, some other experimental stuff. And SkipLayerGuidance for SD3.

---
Also, Frequency-Decoupled Guidance: https://arxiv.org/pdf/2506.19713.

>[!NOTE]
>Requires `kornia>=0.6.8`. Forge default is v0.6.7. Edit `requirements_versions.txt` in the Forge webUI directory.
>
>FDG Option will be hidden if using older version.

Input `Frequency-Decoupled Guidance scaler` is at least two numbers for high frequency and low frequency components:
* `7.0, 1.5` is equivalent to high frequency CFG 7.0 and low frequency CFG 1.5.
* more levels can be used: `7.0, 5.0, 1.5` for high, middle, low
* prefix `*` means the value is a multipler for CFG scale: `*1.1, 1.7` means 1.1*CFG scale for high frequency, and low frequency set to 1.7

---
Also, Tangential Damping https://arxiv.org/pdf/2503.18137
