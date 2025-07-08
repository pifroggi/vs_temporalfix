

















# Add Temporal Coherence to Single Image AI Upscaling Models in Vapoursynth
Also known as temporal consistency, line wiggle fix, stabilization, deshimmering, temporal denoising, or temporal fix.  
This runs on the CPU in parallel to the upscaling on the GPU. Intended for animation.

Check out hddvddegogo's comparisons [here](https://www.youtube.com/watch?v=BXc_Uddt2KA) and [here](https://www.youtube.com/watch?v=u6LHR9_m5rg).



<p align="center">
    <img src="README_img1.gif"/>
</p>

<br />

## Requirements
* [fftw3.3](http://www.fftw.org/download.html) *(required by mvtools)*  
    __Windows__: download and put `.dll` files in plugin folder or add to windows PATH  
    __Linux__: `apt install libfftw3-3 libfftw3-dev` or compile from source
* [mvtools](https://github.com/dubhater/vapoursynth-mvtools) *(release v24 or newer)*
* [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) *(optional, only for tr > 6)*
* [motionmask](https://github.com/dubhater/vapoursynth-motionmask)
* [fillborders](https://github.com/dubhater/vapoursynth-fillborders)
* [zsmooth](https://github.com/adworacz/zsmooth)
* [retinex](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex)
* [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny) *(release r14 or newer)*
* [vszip](https://github.com/dnjulek/vapoursynth-zip) *(optional, slight speed boost)*

## Setup
Put the `vs_temporalfix.py` file into your vapoursynth scripts folder.  
Or install via pip: `pip install git+https://github.com/pifroggi/vs_temporalfix.git`

<br />

## Usage

```python
from vs_temporalfix import vs_temporalfix
clip = vs_temporalfix(clip, strength=400, tr=6, denoise=False, exclude=None, debug=False)
```

__*`clip`*__  
Temporally unstable upscaled clip. Should have no black borders.

__*`strength`*__  
Suppression strength of temporal inconsistencies. Higher means more aggressive. 400-700 works great in most cases.  
The best way to finetune is to find a static scene and adjust till lines and details are stable.  
Reduce if you get blending/ghosting on small movements, especially in dark or hazy scenes.

__*`tr`*__  
Temporal radius sets the number of frames to average over.  
Higher means more stable, especially on slow pans and zooms, but is slower. 6 works great in most cases.  
The best way to finetune is to find a slow pan or zoom and adjust till lines and details are stable.

__*`denoise`* (optional)__  
Removes grain and low frequency noise/flicker left over by the main processing step. Only enable if these issues actually exist! It risks to remove some details like every denoiser, but is useful if you're planning to denoise anyway and has the benefit of almost no performance impact compared to using an additional denoising filter.

__*`exclude`* (optional)__  
Optionally exclude scenes with intended temporal inconsistencies, or in case this causes unexpected issues.  
Example setting 3 scenes: `exclude="[10 20] [600 900] [2000 2500]"`  
First number in the brackets is the first frame of the scene, the second number is the last frame (inclusive).

__*`debug`* (optional)__  
Shows areas that will be left untouched in pink. This includes areas with high motion, scene changes and previously excluded scenes. May help while tuning parameters to see if the area is even affected.

> [!CAUTION]
> * If fps are much lower than the benchmarks, try adding `core.max_cache_size = 15000` (15GB) to your vapoursynth script to allow higher RAM usage. High tr and resolution or large filter scripts may need more.

> [!TIP]
> * Crop any black borders on the input clip, as those may cause ghosting on bright frames.
> * There is a big drop in performance for tr > 6, due to switching from mvtools to mvtools-sf, which is slower.
> * mvtools-sf release r9 and the r10 pre-release are both supported, but r9 is faster for me.

<br />

## Benchmarks

| Hardware    | Resolution | TR | Average FPS
| ----------- | ---------- | -- | -----------
| Ryzen 5900X | 1440x1080  | 6  | ~8 fps
| Ryzen 5900X | 2880x2160  | 6  | ~5 fps

## Alternative Usage Options
Several projects integrated this script to simplify usage without the need for Vapoursynth knowledge.
* __[py_temporalfix](https://github.com/JepEtau/py_temporalfix)__ (Windows only)  
  Simple portable command line tool to just do the temporal fix.
* __[Hybrid](https://www.selur.de/)__ (Windows only)  
  Video filter toolbox with a GUI. Can be a bit overwhelming due to the amount of features, but can upscale and do the temporal fix at the same time, as well as many more filters.
* __[VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)__ (Windows and Linux)  
  Command line AI upscale and interpolation toolbox. Rudimentary knowledge of Docker and Vapoursynth is recommended, but the readme also explains it. Can upscale and do the temporal fix at the same time.
