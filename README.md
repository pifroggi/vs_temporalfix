

















# Add Temporal Coherence to AI Upscales in VapourSynth
When using SISR models (single image super resolution) on video, they tend to create slightly different results each frame. Temporalfix is a post filter that averages these results over multiple frames, removing temporal inconsistencies like fizzle or wiggly lines. Also known as stabilization, deshimmering, temporal denoising, or temporal fix.

<br />

https://github.com/user-attachments/assets/f48545d6-2850-456c-84f4-718339bb3c63

### Requirements

<details>
<summary>Temporalfix AI Model</summary>

If you only intend to use the TensorRT backend, you don't need to install the requirements for the CUDA/CPU backend, and vice versa.
* __TensorRT backend__:
  * [vs-mlrt with TensorRT](https://github.com/AmusementClub/vs-mlrt)
* __CUDA/CPU backend__:
  * [PyTorch with CUDA](https://pytorch.org/)
  * `pip install numpy`

<br />

</details>
<details>
<summary>Temporalfix Classic</summary>

If you only intend to use the Temporalfix AI Model, you don't need to install the requirements for Temporalfix Classic, and vice versa.
* __Required__:  
    * [mvtools](https://github.com/dubhatervapoursynth/vapoursynth-mvtools) *(release v24 or newer)*
    * [motionmask](https://github.com/dubhatervapoursynth/vapoursynth-motionmask)
    * [fillborders](https://github.com/dubhatervapoursynth/vapoursynth-fillborders)
    * [zsmooth](https://github.com/adworacz/zsmooth) *(release v0.14 or newer)*
    * [retinex](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex)
    * [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny) *(release r14 or newer)*
* __Optional:__  
    * [vszip](https://github.com/dnjulek/vapoursynth-zip) *(small speed boost)*
    * [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) *(only needed for tr > 6)*
    * [fftw3.3](http://www.fftw.org/download.html) *(only needed for tr > 6)*  
        __Windows__: download and put `.dll` files in plugin folder next to mvtools-sf  
        __Linux__: via package manager e.g. `apt install libfftw3-dev` or compile from source

<br />

</details>


### Setup
Install/update via pip: `pip install -U git+https://github.com/pifroggi/vs_temporalfix.git`  
Or put the entire `vs_temporalfix` folder into your vapoursynth scripts folder.

<br />

## Temporalfix AI Model
The newest and most capable version of temporalfix. It is easy to use and can run very fast on Nvidia GPUs.

```python
import vs_temporalfix
clip = vs_temporalfix.model(clip, strength=2, tiles=1, backend="tensorrt", num_streams=1, engine_folder=None, exclude=None)
```

__*`clip`*__  
Temporally unstable upscaled clip.

__*`strength`*__  
Suppression strength of temporal inconsistencies in the range 1-3. Higher means more aggressive.  
Higher resolution tends to need higher strength. Too high may oversmooth small movements.

__*`tiles`* (optional)__  
A higher amount of tiles will reduce VRAM usage at the cost of speed.  
This should only be needed on low end hardware. Default tiles=1 uses the full frame, which is fastest.

__*`backend`* (optional)__  
The backend used to run the model:
* `cpu` CPU mode using PyTorch *(very slow)*.
* `cuda` GPU mode using PyTorch with CUDA support. Requires any Nvidia GPU *(fast)*.
* `tensorrt` GPU mode using vs-mlrt with TensorRT support. Requires an Nvidia RTX GPU. On the first run, this mode will automatically build an engine, which may take a few minutes. Changing strength or input dimensions will trigger rebuilding, but previously build engines are stored *(very fast)*.

__*`num_streams`* (optional)__  
Number of parallel TensorRT streams. For high end GPUs higher can be faster, but requires more VRAM. Only effects the TensorRT backend.

__*`engine_folder`* (optional)__  
Optional path to the TensorRT engine storage location. By default engines are stored in `vs_temporalfix/engines`. Only effects the TensorRT backend.

__*`exclude`* (optional)__  
Optionally exclude scenes with intended temporal inconsistencies.  
Brackets define excluded frame ranges. Example for two scenes: `exclude="[10 20] [600 900]"`

> [!TIP]
> Feedback is much appreciated. If the model does not work well for you or causes issues, feel free to open an issue, or contact me via Discord (pifroggi or tepete) and provide a sample. That will help improve it over time.

<br />

## Temporalfix Classic
The original CPU based version. It is slower, harder to use, may miss some areas, and only works for 2D animation.

```python
core.max_cache_size = 15000  # Add near top of vapoursynth script to increase frame cache, else temporalfix will be slow. High tr and resolution, or large filter scripts may need more.
import vs_temporalfix
clip = vs_temporalfix.classic(clip, strength=500, tr=6, denoise=False, exclude=None, debug=False)
```

__*`clip`*__  
Temporally unstable upscaled clip.

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
Optionally exclude scenes with intended temporal inconsistencies.  
Brackets define excluded frame ranges. Example for two scenes: `exclude="[10 20] [600 900]"`

__*`debug`* (optional)__  
Shows areas that will not be fixed in pink. This includes areas with high motion, scene changes and excluded scenes. Can help while tuning parameters to see if the area is even affected.

> [!TIP]
> * Crop any black borders on the input clip, in temporalfix classic those may cause ghosting on bright frames.
> * There is a big drop in performance for tr > 6, due to switching from mvtools to mvtools-sf, which is slower.
> * The plugin [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) release r9 and the r10 pre-release are both supported, but r9 is slightly faster for me.
> * The plugin [zsmooth](https://github.com/adworacz/zsmooth) requires a CPU with AVX2 support (roughly post 2014). If your CPU does not have support, remove zsmooth and replace it with the slightly slower fallbacks [temporalmedian](https://github.com/dubhater/vapoursynth-temporalmedian), [ctmf](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CTMF), and [rgvs](https://github.com/vapoursynth/vs-removegrain).

<br />

## Benchmarks
Model benchmarks were done on a RTX 4090 GPU and Classic benchmarks on a Ryzen 5900X CPU.

| Resolution   | Model (TensorRT) | Model (CUDA) | Classic (with tr=6)
| ------------ | -------- | -------- | -------
| 720x480      | ~320 fps | ~70 fps  | ~35 fps
| 1440x1080    | ~80 fps  | ~32 fps  | ~8 fps
| 2880x2160    | ~20 fps  | ~8 fps   | ~5 fps

<br />

## Third-Party Integrations
Several projects integrated temporalfix to simplify usage without the need for vapoursynth knowledge. Feel free to contact me if you want to be part of this list.
* __[Vapourkit](https://github.com/Kim2091/vapourkit) (Windows only)__  
  Video filter and upscaling program with an easy GUI. This is the easiest way to use it. Just go to plugins, click install, then click add filter and add Temporalfix.
* __[Hybrid](https://www.selur.de/) (Temporalfix Classic only, Windows and Linux)__  
  Video filter toolbox with a GUI. Can be a bit overwhelming due to the amount of features and filters, but Temporalfix Classic is one of them.
* __[py_temporalfix](https://github.com/JepEtau/py_temporalfix) (Temporalfix Classic only, Windows only)__  
  Simple portable command line tool to just run Temporalfix Classic. Easy if you are okay with command line.
* __[VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker) (Temporalfix Classic only, Windows and Linux)__  
  Command line AI upscale and interpolation toolbox that comes with Temporalfix Classic. Rudimentary knowledge of Docker and VapourSynth is recommended, but the readme also explains it.
* __[mpv-cHiDeNoise-AI](https://github.com/animeojisan/mpv-cHiDeNoise-AI) (Temporalfix Classic only, Windows only)__  
  AI upscaling video player based on mpv, which includes a modified lighter version of Temporalfix Classic intended for real-time playback. Mainly intended for japanese audience.
