

















# Add Temporal Coherence to AI Upscales in VapourSynth
When using SISR models (single image super resolution) on video, they tend to create slightly different results each frame. Temporalfix is a post filter that averages these results over multiple frames, removing temporal inconsistencies like fizzle or wiggly lines. Also known as stabilization, deshimmering, temporal denoising, or temporal fix.

<br />

https://github.com/user-attachments/assets/13f05267-cd61-4de9-ad7f-ce8030102465

<br />

## Installation

```
pip install -U vs_temporalfix
```
Optional Extras:
* __Temporalfix AI Model:__ For the CPU/CUDA backends, install [PyTorch with CUDA](https://pytorch.org/).  
* __Temporalfix Classic:__ For tr > 6 support, install [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) and [FFTW 3.3](http://www.fftw.org/download.html) to the vapoursynth plugin directory.

<br />

## Temporalfix AI Model
The newest and most capable version of temporalfix. It is easy to use and can run very fast on Nvidia GPUs.

```python
import vs_temporalfix
clip = vs_temporalfix.model(clip, strength=2.0, tiles=1, backend="tensorrt", num_streams=1, engine_folder=None, exclude=None)
```

__*`clip`*__  
Temporally unstable upscaled clip. Must be in RGB format.

__*`strength`*__  
Suppression strength of temporal inconsistencies in the 0.0-3.0 range. Higher means more aggressive.  
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
The original CPU based version. It is slower, harder to use, may miss some areas, and only works well for 2D animation.

```python
core.max_cache_size = 15000  # Add near top of vapoursynth script to increase frame cache, else temporalfix will be slow. High tr and resolution, or large filter scripts may need more.
import vs_temporalfix
clip = vs_temporalfix.classic(clip, strength=500, tr=6, denoise=False, exclude=None, debug=False)
```

__*`clip`*__  
Temporally unstable upscaled clip. Any format.

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
> * Crop any black borders on the input clip! In temporalfix classic they can cause ghosting on bright frames.
> * There is a big drop in performance for tr > 6, due to switching from mvtools to mvtools-sf, which is slower.

<br />

## Benchmarks
Model benchmarks were done on a RTX 4090 GPU and Classic benchmarks on a Ryzen 5900X CPU.

<table>
  <tr>
    <td valign="top">

<table>
  <thead>
    <tr>
      <th colspan="3">AI Model</th>
    </tr>
    <tr>
      <th>Resolution</th>
      <th>TensorRT</th>
      <th>CUDA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>720x480</td>
      <td>~320 fps</td>
      <td>~70 fps</td>
    </tr>
    <tr>
      <td>1440x1080</td>
      <td>~80 fps</td>
      <td>~32 fps</td>
    </tr>
    <tr>
      <td>2880x2160</td>
      <td>~20 fps</td>
      <td>~8 fps</td>
    </tr>
  </tbody>
</table>

</td>
<td valign="top">

<table>
  <thead>
    <tr>
      <th colspan="4">Classic</th>
    </tr>
    <tr>
      <th>Resolution</th>
      <th>YUV444</th>
      <th>YUV420</th>
      <th>GRAY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>720x480</td>
      <td>~45 fps</td>
      <td>~60 fps</td>
      <td>~85 fps</td>
    </tr>
    <tr>
      <td>1440x1080</td>
      <td>~10 fps</td>
      <td>~14 fps</td>
      <td>~20 fps</td>
    </tr>
    <tr>
      <td>2880x2160</td>
      <td>~5.5 fps</td>
      <td>~8 fps</td>
      <td>~11 fps</td>
    </tr>
  </tbody>
</table>

</td>
  </tr>
</table>

<br />

## Third-Party Integrations
Several projects integrated temporalfix to simplify usage without the need for vapoursynth knowledge. Feel free to contact me if you want to be part of this list.
* __[Vapourkit](https://github.com/Kim2091/vapourkit) (Windows only)__  
  Video filter and upscaling program with an easy GUI. This is the easiest way to use it. Just click on add filter and add one of the two Temporalfix versions.
* __[Hybrid](https://www.selur.de/) (Windows and Linux)__  
  Video filter toolbox with a GUI. Can be a bit overwhelming due to the amount of features and filters, but Temporalfix is one of them.
* __[VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker) (Windows and Linux)__  
  Command line AI upscale and interpolation toolbox that comes with both versions of Temporalfix. Rudimentary knowledge of Docker and VapourSynth is recommended, but the readme also explains it.
* __[mpv-cHiDeNoise-AI](https://github.com/animeojisan/mpv-cHiDeNoise-AI) (Windows only)__  
  AI upscaling video player based on mpv, which includes the Temporalfix AI Models and a lighter modified version of Temporalfix Classic for real-time playback. Mainly intended for japanese audience.
