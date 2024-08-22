

















# Add Temporal Coherence to Single Image AI Upscaling Models in Vapoursynth
Also known as temporal consistency, line wiggle fix, stabilization, temporal denoising, or temporal fix.  
Intended for animation.

This will not add extra work to the upscaling on the GPU and instead run in parallel on the CPU.  

<p align="center">
    <img src="README_example.gif"/>
</p>

<br />

## Requirements
* [fftw3.3](http://www.fftw.org/download.html) (required by mvtools)  
    __Windows__: download and add dll to PATH  
    __Linux__: compile from source or `apt install libfftw3-3`
* [mvtools](https://github.com/dubhater/vapoursynth-mvtools) (release v24 or newer)
* [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) (optional, only for tr > 6)
* [temporalmedian](https://github.com/dubhater/vapoursynth-temporalmedian)
* [motionmask](https://github.com/dubhater/vapoursynth-motionmask)
* [fillborders](https://github.com/dubhater/vapoursynth-fillborders)
* [retinex](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex)
* [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny) (release r14 or newer)
* [ctmf](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF)
* [rgvs](https://github.com/vapoursynth/vs-removegrain)

## Usage

    from vs_temporalfix import vs_temporalfix
    clip = vs_temporalfix(clip, strength=400, tr=6, exclude="[10 20]", debug=False)

__*`clip`*__  
Temporally unstable clip. Should have no black borders.  
Must be in YUV or GRAY format. Full range (PC) input is recommended.

__*`strength`*__  
Suppression strength of temporal inconsistencies. Higher means more aggressive. No influence on processing speed.  
The best way to check is to find a static scene and increase this till details, lines and textures are stable.  
400 works great in many cases. If you get ghosting/blending on small movements or blocky artifacts, reduce this.

__*`tr`*__  
Temporal radius sets the number of frames to average over. Higher means more stable. Influences processing speed.  
The best way to check is to find a slow pan or zoom and increase this till details, lines and textures are stable.  
6 works great in many cases. There is no downside to increasing this further, other than speed and RAM usage.

__*`exclude`* (optional)__  
Optionally exclude scenes with intended temporal inconsistencies (like TV noise), or in case this doesn't work.  
Example setting 3 scenes: `exclude="[10 20] [600 900] [2000 2500]"`  
First number in the brackets is the first frame of the scene, the second number is the last frame (inclusive).

__*`debug`* (optional)__  
Shows protected areas, scene changes and exclusions in pink half transparent on top of the clip.
Protected areas have motions that are large enough to exclude from processing. This avoids blending and ghosting.

## Tips & Troubleshooting
* Make sure to check very dark, hazy, or faint scenes for ghosting/blending and reduce strength if necessary.
* If fps are much lower than the benchmarks would suggest, try increasing Vapoursynth's RAM cache by adding `core.max_cache_size = 20000` (20GB, adjust if needed) near the top of your script. RAM requirements depend on tr and resolution.
* There is a big drop in performance for tr > 6, due to switching from mvtools to mvtools-sf, which is slower.
* mvtools-sf release r9 and the r10 pre-release will both work, but r9 is faster for me.

## Benchmarks

| Hardware    | Resolution | TR | Average FPS
| ----------- | ---------- | -- | -----------
| Ryzen 5900X | 1440x1080  | 6  | ~7 fps
| Ryzen 5900X | 2880x2160  | 6  | ~4 fps
