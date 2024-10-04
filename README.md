## Adaptive Projected Guidance ##
### APG code from ELIMINATING OVERSATURATION AND ARTIFACTS OF HIGH GUIDANCE SCALES IN DIFFUSION MODELS ###
### credit to Seyedmorteza Sadat, Otmar Hilliges, Romann M. Weber ###
### https://arxiv.org/pdf/2410.02416 ###
---
controls described in the paper

eta affects saturation

rescale threshold ... highly variable depending on model, experiment

momentum between -0.75 and -0.25 recommended

---
The paper mostly compares APG with too-high CFG, which I'm not sure is all that useful - we already know high CFG burns badly, so a better comparison would be between APG and reasonable CFG to see if we really do get better quality, better prompt adherence, etc.
Anyway, it does *something*.

#### sdxl model haveall cfg3.3 ####
![](lavender2.png) 

#### sd1.5 model swizz8real cfg 15 ####
![](woodelves2.png) 
