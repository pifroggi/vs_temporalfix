

















# Add Temporal Coherence to Single Image AI Upscaling Models in Vapoursynth
Also known as temporal consistency, line wiggle fix, stabilization, temporal denoising, or temporal fix.  
This will not add extra work to the upscaling on the GPU and instead run in parallel on the CPU.  
Intended for animation.

__Comparisons (by hddvddegogo):__  
https://www.youtube.com/watch?v=BXc_Uddt2KA  
https://www.youtube.com/watch?v=u6LHR9_m5rg  

<p align="center">
    <img src="README_example.gif"/>
</p>

<br />

## Requirements
* [fftw3.3](http://www.fftw.org/download.html) (required by mvtools)  
    __Windows__: download and put `.dll` files in plugin folder or add to windows PATH  
    __Linux__: `apt install libfftw3-3 libfftw3-dev` or compile from source
* [mvtools](https://github.com/dubhater/vapoursynth-mvtools) (release v24 or newer)
* [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) (optional, only for tr > 6)
* [temporalmedian](https://github.com/dubhater/vapoursynth-temporalmedian)
* [motionmask](https://github.com/dubhater/vapoursynth-motionmask)
* [fillborders](https://github.com/dubhater/vapoursynth-fillborders)
* [retinex](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex)
* [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny) (release r14 or newer)
* [ctmf](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF)
* [rgvs](https://github.com/vapoursynth/vs-removegrain)

## Setup
Put the `vs_temporalfix.py` file into your vapoursynth scripts folder.  
Or install via pip: `pip install git+https://github.com/pifroggi/vs_temporalfix.git`

<br />

## Usage

    from vs_temporalfix import vs_temporalfix
    clip = vs_temporalfix(clip, strength=400, tr=6, exclude="[10 20]", debug=False)

__*`clip`*__  
Temporally unstable clip. Must be in YUV or GRAY format.  
Should have no black borders. Full range (PC) input is recommended.

__*`strength`*__  
Suppression strength of temporal inconsistencies. Higher means more aggressive. 400 works great in most cases.  
The best way to finetune is to find a static scene and increase this till lines and details are stable.  
Reduce if you get blending/ghosting on small movements, especially in dark or hazy scenes, or blocky artifacts.

__*`tr`*__  
Temporal radius sets the number of frames to average over.  
Higher means more stable, especially on slow pans and zooms, but makes it slower. 6 works great in most cases.  
The best way to finetune is to find a slow pan or zoom and increase this till lines and details are stable.

__*`exclude`* (optional)__  
Optionally exclude scenes with intended temporal inconsistencies (like TV noise), or in case this doesn't work.  
Example setting 3 scenes: `exclude="[10 20] [600 900] [2000 2500]"`  
First number in the brackets is the first frame of the scene, the second number is the last frame (inclusive).

__*`debug`* (optional)__  
Shows protected areas that will be left untouched in pink. This includes areas with high motion, scene changes and excluded scenes. May help while tuning parameters to see if the area is even affected.

<br />

## Tips & Troubleshooting
> [!CAUTION]
> * If fps are much lower than the benchmarks, try adding `core.max_cache_size = 20000` (20GB) near the top of your vapoursynth script to allow higher RAM usage. For higher tr or resolution, increase further if needed.

> [!TIP]
> * There is a big drop in performance for tr > 6, due to switching from mvtools to mvtools-sf, which is slower.
> * mvtools-sf release r9 and the r10 pre-release are both supported, but r9 is faster for me.

## Benchmarks

| Hardware    | Resolution | TR | Average FPS
| ----------- | ---------- | -- | -----------
| Ryzen 5900X | 1440x1080  | 6  | ~7 fps
| Ryzen 5900X | 2880x2160  | 6  | ~4 fps

<br />

## Alternative Usage Options
Several projects integrated this script to simplify usage without Vapoursynth knowledge.
* __[py_temporalfix](https://github.com/JepEtau/py_temporalfix)__ (Windows only)  
  Simple portable command line tool to just do the temporal fix.
* __[Hybrid](https://www.selur.de/)__ (Windows only)  
  Video filter toolbox with a GUI. Can be a bit overwhelming due to the amount of features, but can upscale and do the temporal fix at the same time, as well as many more filters.
* __[VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)__ (Windows and Linux)  
  Command line AI upscale and interpolation toolbox. Rudimentary knowledge of Docker and Vapoursynth is recommended, but the readme also explains it. Can upscale and do the temporal fix at the same time.
  
