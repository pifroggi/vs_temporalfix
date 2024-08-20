


















# Vapoursynth function to add Temporal Coherence to AI Upscales
Also known as temporal consistency, stabilization, line wiggle removal, remove dancing details, temporal denoising, or temporal fix.  
This will not add extra work to the upscaling on the GPU and instead run in parallel on the CPU.  
Intended for animation.

<p align="center">
    <img src="README_example.gif"/>
</p>

<br />

## Requirements
* [fftw3.3](http://www.fftw.org/download.html) (required by mvtools)  
__Windows__: download and add dll to PATH  
__Linux__: compile from source or `apt install libfftw3-3`
* [mvtools](https://github.com/dubhater/vapoursynth-mvtools) (release r24 or newer)
* [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) (optional, only for radius > 6)
* [temporalmedian](https://github.com/dubhater/vapoursynth-temporalmedian)
* [motionmask](https://github.com/dubhater/vapoursynth-motionmask)
* [fillborders](https://github.com/dubhater/vapoursynth-fillborders)
* [retinex](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex)
* [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny) (release r14 or newer)
* [ctmf](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF)
* [rgvs](https://github.com/vapoursynth/vs-removegrain)

## Usage

    from vs_temporalfix import vs_temporalfix
    clip = vs_temporalfix(clip, strength=400, tr=6, exclude=None, debug=False)

__*`clip`*__  
Temporally unstable clip. Should have no black borders.  
Must be in YUV or GRAY format. Full range (PC) input is recommended.

__*`strength`*__  
Suppression strength of temporal inconsistencies. Higher means more aggressive. Has no influence on processing speed.  
400 works great in many cases.  
The best way to check is to find a static scene and increase this till details, lines and textures are all stable.  
If you get ghosting/blending on small movements or blocky artifacts, reduce this.

__*`tr`*__  
Temporal radius sets how many frames will be included in the calculation. Higher means more stable. This influences processing speed.  
6 works great in many cases.  
The best way to check is to find a slow pan or zoom and increase this till details, lines and textures are all stable.  
There is no downside to increasing this further, other than processing speed and RAM usage.

__*`exclusion`* (optional)__  
While this is often not needed, sometimes there are scenes with intended temporal inconsistencies (like TV static noise), or this just doesn't want to work. Set them here.  
Example setting 3 scenes: `exclude="[100 400] [600 900] [2000 2500]"` First number is the first frame of the scene, second number is the last frame (inclusive).

__*`debug`* (optional)__  
Shows protected regions, scene changes and exclusions in pink (white if clip is gray) half transparent on top of the clip.

## Tips
* Make sure to check very dark, hazy, or faint scenes for ghosting/blending and reduce strength if necessary.
* If your fps numbers are vastly slower than the benchmarks would suggest, try allowing vapoursynth to use more RAM by adding this near the top of you script `core.max_cache_size = 20000` (20gb, adjust as needed). RAM requirements are influenced by tr and resolution.
* There is a big drop in performance for tr > 6, due to switching from mvtools to mvtools-sf, which is slower.
* mvtools-sf release r9 and the r10 pre-release will both work, but r9 is faster for me.

## Benchmarks

| Hardware    | Resolution        | tr | Average FPS
| ----------- | ----------------- | -- | -----------        
| Ryzen 5900X | 1440x1080 (1080p) | 6  | ~7 fps
| Ryzen 5900X | 2880x2160 (4k)    | 6  | ~4 fps
