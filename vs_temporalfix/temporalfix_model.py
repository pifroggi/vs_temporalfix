
# Script, architecture, and model training by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

# Frame alignment and masking idea from RIFE https://github.com/hzwer/ECCV2022-RIFE

import os
import shutil
import logging
import subprocess
import vapoursynth as vs
from pathlib import Path
from .utils import gen_shifts, get_tiles, get_spans, exclude_regions

core = vs.core


def _pytorch(clip, strength=2, tiles=1, device="cuda", exclude=None):
    import torch
    import threading
    import numpy as np
    from collections import OrderedDict
    from .models.temporalfix_arch import temporalfix_arch
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_temporalfix: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_temporalfix: Clip must have constant format and dimensions.")
    if clip.num_frames < 2:
        raise ValueError("vs_temporalfix: Clip must be at least 2 frames long.")
    if clip.format.color_family != vs.RGB:
        raise ValueError("vs_temporalfix: Clip must be in RGB format.")

    orig_clip   = clip
    clip_w      = clip.width
    clip_h      = clip.height
    orig_format = clip.format.id
    not_tiled   = tiles == 1
    overlap     = min(64, int(64 * ((clip_w * clip_h) / (1920 * 1080)) ** 0.5))  # overlap gets smaller for smaller inputs
    device      = torch.device(device)
    use_cuda    = device.type == "cuda"
    fp16        = use_cuda and torch.cuda.get_device_capability(device=device)[0] >= 7
    req_format  = vs.RGBH if fp16 else vs.RGBS
    dtype_np    =    np.float16 if fp16 else    np.float32
    dtype_torch = torch.float16 if fp16 else torch.float32

    # select tiling
    tile_w, tile_h, cols, rows = get_tiles(clip_w=clip_w, clip_h=clip_h, tiles=tiles, overlap=overlap)
    x_spans = get_spans(clip_w, tile_w, cols)
    y_spans = get_spans(clip_h, tile_h, rows)
    tile_spans = []
    for tile_y, (y0, y1, dst_y0, dst_y1) in enumerate(y_spans):
        for tile_x, (x0, x1, dst_x0, dst_x1) in enumerate(x_spans):
            tile_idx = tile_y * cols + tile_x
            tile_spans.append((tile_idx, x0, x1, y0, y1, dst_x0, dst_x1, dst_y0, dst_y1))

    # select model
    if strength == 1:
        model_file = "temporalfix_s1_v1.pth"
    elif strength == 2:
        model_file = "temporalfix_s2_v1.pth"
    elif strength == 3:
        model_file = "temporalfix_s3_v1.pth"
    else:
        raise ValueError("vs_temporalfix: Strength must be in the range 1-3.")

    # load model
    current_dir = os.path.dirname(__file__)
    model_path  = os.path.join(current_dir, "models", model_file)
    model = temporalfix_arch(fixed_hw=(tile_h, tile_w), conf_thresh=0.6, min_support=1, gate_slope=12.0, count_slope=4.0)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if fp16:
        model.half()

    class LRUFrameCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.store = OrderedDict()
            self.lock = threading.Lock()

        def get(self, key):
            with self.lock:
                value = self.store.pop(key, None)
                if value is not None:
                    self.store[key] = value
                return value

        def put(self, key, value):
            with self.lock:
                if key in self.store:
                    self.store.pop(key)
                elif len(self.store) >= self.capacity:
                    self.store.popitem(last=False)
                self.store[key] = value

    def _frame_to_tensor(frame, frame_idx, tile_idx=None, x0=None, x1=None, y0=None, y1=None):
        # get from cache if possible
        cache_key = frame_idx if tile_idx is None else (frame_idx, tile_idx)
        cached = FRAME_CACHE.get(cache_key)
        if cached is not None:
            return cached

        # else convert
        if tile_idx is None:
            array = np.empty((3, clip_h, clip_w), dtype=dtype_np)
            for p in range(3):
                array[p] = np.asarray(frame[p])
            tensor = torch.from_numpy(array).unsqueeze(0)  # [1, C, H, W]
            tensor = tensor.to(device=device, non_blocking=use_cuda)
        else:
            array = np.empty((3, y1 - y0, x1 - x0), dtype=dtype_np)
            for p in range(3):
                array[p] = np.asarray(frame[p])[y0:y1, x0:x1]
            tensor = torch.from_numpy(array).unsqueeze(0)  # [1, C, H, W]

        FRAME_CACHE.put(cache_key, tensor)
        return tensor

    def _tensor_to_frame(tensor, frame):
        array = tensor.detach()[0]
        if array.device.type != "cpu":
            array = array.cpu()
        array = array.numpy()
        for p in range(3):
            np.copyto(np.asarray(frame[p]), array[p])

    def _process_frame(n, f):
        with torch.inference_mode():
            out = f[3].copy()

            if not_tiled:
                tensors = [_frame_to_tensor(f[i], frame_idx) for i, frame_idx in enumerate(range(n - 3, n + 4))]
                tensor = torch.cat(tensors, dim=0).unsqueeze(0)
                _tensor_to_frame(model(tensor.to(device=device, non_blocking=use_cuda)), out)
                return out

            out_tensor = torch.empty((1, 3, clip_h, clip_w), dtype=dtype_torch)
            for tile_idx, x0, x1, y0, y1, dst_x0, dst_x1, dst_y0, dst_y1 in tile_spans:
                tile_tensors = [_frame_to_tensor(f[i], frame_idx, tile_idx=tile_idx, x0=x0, x1=x1, y0=y0, y1=y1) for i, frame_idx in enumerate(range(n - 3, n + 4))]
                tile_inp = torch.cat(tile_tensors, dim=0).unsqueeze(0)
                tile_out = model(tile_inp.to(device=device, non_blocking=use_cuda))[0]  # [C, tile_h, tile_w]
                out_tensor[0, :, dst_y0:dst_y1, dst_x0:dst_x1] = tile_out[:, dst_y0 - y0:dst_y1 - y0, dst_x0 - x0:dst_x1 - x0]

            _tensor_to_frame(out_tensor, out)
            return out
    
    # clamp and convert
    if clip.format.sample_type == vs.FLOAT:
        clip = core.std.Expr(clip, expr=["x 0 max 1 min"])
    if clip.format.id != req_format:
        clip = core.resize.Point(clip, format=req_format)

    # shift and inference
    input_clips = gen_shifts(clip, radius=3)          # [-3, -2, -1, 0, +1, +2, +3]
    FRAME_CACHE = LRUFrameCache(capacity=10 * tiles)  # cache converted frame tensors
    out = core.std.ModifyFrame(clip, clips=input_clips, selector=_process_frame)

    # convert back and return
    if out.format.id != orig_format:
        return core.resize.Point(out, format=orig_format)
    return exclude_regions(out, orig_clip, exclude=exclude)  # exclude regions from temporalfix


def _get_trtexec():
    # first search for tensorrt plugins, then check for trtexec
    exe_name = "trtexec.exe" if os.name == "nt" else "trtexec"
    plugins_path = None

    try:
        info = core.trt.Version()
    except Exception as e:
        raise RuntimeError("vs_temporalfix: Please install a version of vs-mlrt with TensorRT support.") from e

    path = info.get("path")

    # get plugin path
    if isinstance(path, bytes):
        path = path.decode(errors="ignore")
    if path:
        plugins_path = os.path.dirname(path)

    # try finding vsmlrt trtexec first, then check for system trtexec
    if plugins_path is not None:
        local_trtexec = Path(plugins_path) / "vsmlrt-cuda" / exe_name
        if local_trtexec.is_file() and os.access(str(local_trtexec), os.X_OK):
            return local_trtexec

    system_trtexec = shutil.which("trtexec")
    if system_trtexec is not None:
        return Path(system_trtexec)

    raise FileNotFoundError("vs_temporalfix: trtexec not found. Please install a version of vs-mlrt with TensorRT support. Make sure to follow the installation instructions.")


def _get_engine(onnx_path, engine_dir, model_name, engine_w, engine_h, opt_level=3, force_rebuild=False) -> str:
    # build or get path to tensorrt engine
    os.makedirs(engine_dir, exist_ok=True)  # create engine folder if needed
    engine_name  = f"{model_name}_h{engine_h}_w{engine_w}_fp16.engine"
    engine_path  = os.path.join(engine_dir, engine_name)
    trtexec_path = _get_trtexec()

    # if engine file exist, return it
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path

    # else build new engine
    logging.warning("vs_temporalfix: Building new TensorRT engine for model %s with width=%d and height=%d. This may take a few minutes.", model_name, engine_w, engine_h)
    opt_shapes = f"input:1x21x{engine_h}x{engine_w}"
    cmd = [
        str(trtexec_path),
        "--stronglyTyped",
        "--inputIOFormats=fp16:chw",
        "--outputIOFormats=fp16:chw",
        "--skipInference",
        "--memPoolSize=workspace:4096",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--optShapes={opt_shapes}",
        f"--builderOptimizationLevel={opt_level}",
    ]

    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = (
            "vs_temporalfix: trtexec failed while building the TensorRT engine.\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Return code: {e.returncode}\n"
        )
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e

    logging.warning("vs_temporalfix: Engine building complete.")
    return engine_path


def _tensorrt_inference(input_clips, onnx_path, engine_dir, model_name, clip_w, clip_h, tiles=1, overlap=64, opt_level=3, num_streams=1, force_rebuild=False):
    tile_w, tile_h, _, _ = get_tiles(clip_w=clip_w, clip_h=clip_h, tiles=tiles, overlap=overlap)
    engine_path = _get_engine(onnx_path=onnx_path, engine_dir=engine_dir, model_name=model_name, engine_w=tile_w, engine_h=tile_h, opt_level=opt_level, force_rebuild=force_rebuild)
    model_args  = dict(engine_path=engine_path, num_streams=num_streams, **(dict(tilesize=(tile_w, tile_h), overlap=(overlap, overlap)) if tiles > 1 else {}))

    # try inference, rebuild engine if it fails
    try:
        out = core.trt.Model(input_clips, **model_args)
    except vs.Error as e:
        err_msg = str(e).lower()
        serialization_keywords = ("serialize", "serialization", "deserialize", "deserialization")
        if any(k in err_msg for k in serialization_keywords) and not force_rebuild:
            logging.warning("vs_temporalfix: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
            model_args["engine_path"] = _get_engine(onnx_path=onnx_path, engine_dir=engine_dir, model_name=model_name, engine_w=tile_w, engine_h=tile_h, opt_level=opt_level, force_rebuild=True)
            out = core.trt.Model(input_clips, **model_args)
        else:
            raise
    return out


def _tensorrt(clip, strength=2, tiles=1, num_streams=1, exclude=None):
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_temporalfix: Clip must be a vapoursynth clip.")
    if clip.format.id  == vs.PresetVideoFormat.NONE or clip.width  == 0 or clip.height  == 0:
        raise TypeError("vs_temporalfix: Clip must have constant format and dimensions.")
    if clip.num_frames < 2:
        raise ValueError("vs_temporalfix: Clip must be at least 2 frames long.")
    if clip.format.id not in [vs.RGBH]:
        raise ValueError("vs_temporalfix: Clip must be in RGBH format for TensorRT backend.")
    if num_streams < 1:
        raise ValueError("vs_temporalfix: Number of parallel TensorRT streams (num_streams) must be at least 1.")
    
    # select model
    if strength == 1:
        model_file = "temporalfix_s1_v1_op18_fp16.onnx"
        model_name = "temporalfix_s1_v1"
    elif strength == 2:
        model_file = "temporalfix_s2_v1_op18_fp16.onnx"
        model_name = "temporalfix_s2_v1"
    elif strength == 3:
        model_file = "temporalfix_s3_v1_op18_fp16.onnx"
        model_name = "temporalfix_s3_v1"
    else:
        raise ValueError("vs_temporalfix: Strength must be in the range 1-3.")
    
    clip_w        = clip.width
    clip_h        = clip.height
    opt_level     = 3
    overlap       = min(64, int(64 * ((clip_w * clip_h) / (1920 * 1080)) ** 0.5))  # overlap gets smaller for smaller inputs
    force_rebuild = False
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    onnx_path     = os.path.join(current_dir, "models", model_file)
    engine_dir    = os.path.join(current_dir, "engines")

    # shift and inference
    clip = core.std.Expr(clip, expr=["x 0 max 1 min"])  # clamp
    input_clips = gen_shifts(clip, radius=3)  # [-3, -2, -1, 0, +1, +2, +3]
    out = _tensorrt_inference(input_clips=input_clips, onnx_path=onnx_path, engine_dir=engine_dir, model_name=model_name, clip_w=clip_w, clip_h=clip_h, opt_level=opt_level, tiles=tiles, overlap=overlap, num_streams=num_streams, force_rebuild=force_rebuild)
    out = core.std.CopyFrameProps(out, clip)  # copy props to make sure they are not from the shifted -3 clip
    return exclude_regions(out, clip, exclude=exclude)  # exclude regions from temporalfix


def model(clip, strength=2, tiles=1, backend="tensorrt", num_streams=1, exclude=None):
    """Add temporal coherence to single image AI upscaling models. Also known as temporal consistency, line wiggle fix, stabilization, deshimmering.

    Args:
        clip: Temporally unstable upscaled clip.
        strength: Suppression strength of temporal inconsistencies in the range `1-3`. Higher means more aggressive. 
            Higher resolution tends to need higher strength. Too high may oversmooth small movements.
        tiles: A higher amount of tiles will reduce VRAM usage at the cost of speed. 
            This should only be needed on low end hardware. `tiles=1` will use the full frame, which is fastest.
        backend: The backend used to run the model.
            - `cpu` = CPU mode using PyTorch (slowest).
            - `cuda` = GPU mode using PyTorch with CUDA support. Requires any Nvidia GPU (faster).
            - `tensorrt` = GPU mode using vs-mlrt with TensorRT support. Requires an Nvidia RTX GPU (fastest).
        num_streams: Number of parallel TensorRT streams. For high end GPUs higher can be faster, but requires more VRAM. Only effects the TensorRT backend.
        exclude: Optionally exclude scenes with intended temporal inconsistencies, or in case this causes unexpected issues. 
            Example setting 3 scenes: `exclude="[10 20] [600 900] [2000 2500]"`. 
            First number in the brackets is the first frame of the scene, the second number is the last frame (inclusive).
    """
    
    if backend in ["cpu", "cuda"]:
        return _pytorch(clip, strength=strength, tiles=tiles, device=backend, exclude=exclude)
    if backend in ["tensorrt", "trt"]:
        return _tensorrt(clip, strength=strength, tiles=tiles, num_streams=num_streams, exclude=exclude)
    raise ValueError("vs_temporalfix: Backend must be 'cpu', 'cuda', or 'tensorrt'.")
