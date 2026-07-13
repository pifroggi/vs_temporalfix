
# Script, architecture, and model training by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

import os
import re
import math
import shutil
import logging
import subprocess
import vapoursynth as vs
from pathlib import Path
from .utils import basic_expr, gen_shifts, get_tiles, get_spans, exclude_regions, interpolate_onnx

core = vs.core


def _pytorch(clip, strength=2, tiles=1, device="cuda", exclude=None):
    import threading
    import numpy as np
    from collections import OrderedDict

    if device == "cpu":
        try:
            import torch
        except ImportError:
            raise RuntimeError("vs_temporalfix: The CPU/CUDA backends require PyTorch. Please install it from https://pytorch.org/ or choose a different backend. For the CUDA backend specifically, install a version of PyTorch with CUDA support.") from None

    if device == "cuda":
        try:
            import torch
        except ImportError:
            raise RuntimeError("vs_temporalfix: The CUDA backend requires PyTorch with CUDA. Please install a version of PyTorch with CUDA support from https://pytorch.org/ or choose a different backend.") from None
        if not torch.cuda.is_available():
            raise RuntimeError("vs_temporalfix: The CUDA backend requires PyTorch with CUDA, but the installed version has no CUDA support. Please upgrade to a version with CUDA support from https://pytorch.org/ or choose a different backend.")

    from .models.temporalfix_arch import temporalfix_arch
    os.environ["CUDA_MODULE_LOADING"] = "LAZY"

    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_temporalfix: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_temporalfix: Clip must have constant format and dimensions.")
    if clip.num_frames < 4:
        raise ValueError("vs_temporalfix: Clip must be at least 4 frames long.")
    if clip.format.color_family != vs.RGB:
        raise ValueError("vs_temporalfix: Clip must be in RGB format.")
    if strength < 0 or strength > 3:
        raise ValueError("vs_temporalfix: Strength must be in the 0.0-3.0 range.")

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
    strength       = round(float(strength), 2)
    strength_lower = math.floor(strength)
    strength_upper = math.ceil(strength)
    weighting      = strength - strength_lower
    current_dir    = os.path.dirname(__file__)
    model_files    = {
        0: "temporalfix_s0_v1.pth",
        1: "temporalfix_s1_v1.1.pth",
        2: "temporalfix_s2_v1.pth",
        3: "temporalfix_s3_v1.pth",
    }

    # load model
    def _load_state(model_file):
        model_path = os.path.join(current_dir, "models", model_file)
        return torch.load(model_path, map_location="cpu", weights_only=True)

    if strength == 0:
        return clip
    if strength_lower == strength_upper:  # if both the same, load model directly
        state = _load_state(model_files[strength_lower])
    else:                                 # else load both and interpolate
        state_lower = _load_state(model_files[strength_lower])
        state_upper = _load_state(model_files[strength_upper])
        state = OrderedDict((key, torch.lerp(value, state_upper[key], weighting) if torch.is_floating_point(value) else value) for key, value in state_lower.items())
    
    model = temporalfix_arch(fixed_hw=(tile_h, tile_w), conf_thresh=0.6, min_support=1, gate_slope=12.0, count_slope=4.0)
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

    def _pytorch_inference(n, f):
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
        clip = basic_expr(clip, expr=["x 0 max 1 min"])
    if clip.format.id != req_format:
        clip = core.resize.Point(clip, format=req_format)

    # shift and inference
    input_clips = gen_shifts(clip, radius=3)          # [-3, -2, -1, 0, +1, +2, +3]
    FRAME_CACHE = LRUFrameCache(capacity=10 * tiles)  # cache converted frame tensors
    out = core.std.ModifyFrame(clip, clips=input_clips, selector=_pytorch_inference)

    # convert back and return
    if out.format.id != orig_format:
        out = core.resize.Point(out, format=orig_format)
    return exclude_regions(out, orig_clip, exclude=exclude)  # exclude regions from temporalfix


def _get_builder(plugin_path, trt_version, cuda_major):
    # finds compatible tensorrt engine builders
    exe_name = "trtexec.exe" if os.name == "nt" else "trtexec"
    builders = []
    errors   = []
    
    # check for python tensorrt
    try:
        import tensorrt
        package_version = list(map(int, tensorrt.__version__.split(".")[:3]))
        if package_version == trt_version:
            builders.append(["python", tensorrt])
        else:
            errors.append(f"Python TensorRT: Wrong version {'.'.join(map(str, package_version))}")
    except ImportError:
        errors.append("Python TensorRT: Not found.")
    except Exception:
        errors.append("Python TensorRT: Found but failed to check version.")
    
    # check for bundled trtexec
    bundled_trtexec = Path(plugin_path) / "vsmlrt-cuda" / exe_name
    if bundled_trtexec.is_file() and os.access(str(bundled_trtexec), os.X_OK):
        builders.append(["trtexec", bundled_trtexec])
    else:
        errors.append(f"Bundled trtexec: Not found.")

    # check for system trtexec
    system_trtexec = shutil.which("trtexec")
    if system_trtexec is not None:
        try:
            trtexec_path = Path(system_trtexec)
            help_output  = subprocess.run([str(trtexec_path), "--help"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
            help_output  = f"{help_output.stdout}\n{help_output.stderr}"
            
            trtexec_version = None
            trtexec_version = re.search(r"\[TensorRT v(\d+)\]", help_output)
            if trtexec_version is None:
                raise RuntimeError("vs_temporalfix: Internal Error: Regex failed to find the version.")

            trtexec_version = int(trtexec_version.group(1))
            trtexec_version = [trtexec_version // 10000, (trtexec_version % 10000) // 100, trtexec_version % 100]
            if trtexec_version == trt_version:
                builders.append(["trtexec", trtexec_path])
            else:
                errors.append(f"System trtexec: Wrong version {'.'.join(map(str, trtexec_version))}")
        except Exception:
            errors.append("System trtexec: Found but failed to check version.")
    else:
        errors.append("System trtexec: Not found.")
    
    # return first compatible builder
    if builders:
        return builders[0]
    
    errors = "\n".join(f"{builder}" for builder in errors)
    raise FileNotFoundError(f"vs_temporalfix: No compatible TensorRT engine builder found. Please install the python package 'tensorrt' or install trtexec. The required TensorRT version is {'.'.join(map(str, trt_version))}. The required CUDA version is {cuda_major}.\n{errors}")


def _build_engine_trtexec(onnx_path, engine_path, engine_w, engine_h, trt_version, trtexec_path):
    # build engine using trtexec, supports trt 10 and 11

    # settings
    opt_shapes = f"input:1x21x{engine_h}x{engine_w}"
    io_formats = f"fp16:chw" if trt_version[0] < 11 else "chw"
    cmd = [
        str(trtexec_path),
        *(["--stronglyTyped"] if trt_version[0] < 11 else []),
        "--skipInference",
        "--memPoolSize=workspace:4096",
        "--builderOptimizationLevel=3",
        f"--inputIOFormats={io_formats}",
        f"--outputIOFormats={io_formats}",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--optShapes={opt_shapes}",
    ]

    # build
    try:
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = (
            "vs_temporalfix: Internal Error: trtexec failed while building the TensorRT engine.\n"
            f"  Command: {' '.join(cmd)}\n"
            f"  Return code: {e.returncode}\n"
        )
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e


def _build_engine_python(onnx_path, engine_path, engine_w, engine_h, trt_package):
    # build engine using tensorrt python bindings, supports only trt 11
    trt = trt_package

    # custom logger for errors
    class _TrtLogger(trt.ILogger):
        def __init__(self):
            trt.ILogger.__init__(self)
            self.messages = []
        def log(self, severity, msg):
            if severity <= trt.Logger.WARNING:
                self.messages.append((severity, msg))
        def get_log(self):
            return "\n".join(f"  [{severity}] {msg}" for severity, msg in self.messages)

    # initialize trt and load model
    logger  = _TrtLogger()
    builder = trt.Builder(logger)
    network = builder.create_network()
    config  = builder.create_builder_config()
    parser  = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(f"  {parser.get_error(i)}" for i in range(parser.num_errors))
        raise RuntimeError(f"vs_temporalfix: Internal Error: TensorRT failed while parsing the ONNX model.\n{errors}")
    
    # settings
    opt_shapes = (1, 21, engine_h, engine_w)                                                                          # optShapes
    network.get_input(0).allowed_formats = network.get_output(0).allowed_formats = 1 << int(trt.TensorFormat.LINEAR)  # IOFormats:chw
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 << 20)                                            # workspace:4096
    config.builder_optimization_level = 3                                                                             # builderOptimizationLevel=3

    # build
    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, opt_shapes, opt_shapes, opt_shapes)
    config.add_optimization_profile(profile)
    engine  = builder.build_serialized_network(network, config)
    if engine is None:
        log = logger.get_log()
        msg = "vs_temporalfix: Internal Error: TensorRT failed while building the TensorRT engine."
        if log:
            msg += f"\n=== TensorRT log ===\n{log}"
        raise RuntimeError(msg)
    
    # save engine
    with open(engine_path, "wb") as f:
        f.write(engine)


def _get_engine(model_files, onnx_dir, engine_dir, strength, engine_w, engine_h, force_rebuild=False) -> str:
    # check plugin version
    try:
        info = core.trt.Version()
    except Exception as e:
        raise RuntimeError("vs_temporalfix: Please install a version of vs-mlrt with TensorRT support or choose a different backend.") from e
    
    # select model
    strength_lower = math.floor(strength)
    strength_upper = math.ceil(strength)
    name_strength  = str(strength).rstrip("0").rstrip(".")  # remove extra zeros and dot if not needed

    if strength_lower == strength_upper:  # if both the same, load model directly
        model_file = model_files[strength_lower]
        model_name = os.path.splitext(model_file)[0].split("_op")[0]
        onnx_path  = os.path.join(onnx_dir, model_file)
    else:                                 # else load both and interpolate linearly
        model_file_lower = model_files[strength_lower]
        model_file_upper = model_files[strength_upper]
        name_lower = os.path.splitext(model_file_lower)[0].split("temporalfix_")[1].split("_op")[0]
        name_upper = os.path.splitext(model_file_upper)[0].split("temporalfix_")[1].split("_op")[0]
        model_name = f"temporalfix_s{name_strength}_[{name_lower}+{name_upper}]"
        weighting  = strength - strength_lower
        onnx_path  = [os.path.join(onnx_dir, model_file_lower), os.path.join(onnx_dir, model_file_upper)]
    
    # get path to tensorrt engine
    os.makedirs(engine_dir, exist_ok=True)  # create engine folder if needed
    engine_name  = f"{model_name}_h{engine_h}_w{engine_w}_fp16.engine"
    engine_path  = os.path.join(engine_dir, engine_name)
    temp_dir     = None
    
    # if engine file exist, return it
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path
    
    # get plugin info
    plugin_path = os.path.dirname(info["path"].decode(errors="ignore"))
    trt_version = int(info["tensorrt_version"].decode(errors="ignore"))
    trt_version = [trt_version // 10000, (trt_version % 10000) // 100, trt_version % 100]
    cuda_major  = int(info["cuda_runtime_version"].decode(errors="ignore")) // 1000
    
    # interpolate onnx if needed
    if isinstance(onnx_path, list):
        import tempfile
        temp_dir  = tempfile.TemporaryDirectory(prefix=f"{model_name}_", dir=engine_dir)
        temp_path = os.path.join(temp_dir.name, f"{model_name}.onnx")
        interpolate_onnx(onnx_path_lower=onnx_path[0], onnx_path_upper=onnx_path[1], save_path=temp_path, weighting=weighting)
        onnx_path = temp_path
    
    # build new engine
    logging.warning("vs_temporalfix: Building new TensorRT engine for strength=%s with width=%d and height=%d. This may take a few minutes.", name_strength, engine_w, engine_h)
    try:
        builder_info = _get_builder(plugin_path=plugin_path, trt_version=trt_version, cuda_major=cuda_major)
        if builder_info[0] == "python":
            _build_engine_python(onnx_path=onnx_path, engine_path=engine_path, engine_w=engine_w, engine_h=engine_h, trt_package=builder_info[1])
        elif builder_info[0] == "trtexec":
            _build_engine_trtexec(onnx_path=onnx_path, engine_path=engine_path, engine_w=engine_w, engine_h=engine_h, trt_version=trt_version, trtexec_path=builder_info[1])
        else:
            raise RuntimeError(f"vs_temporalfix: Internal Error: Unknown TensorRT engine builder: {builder_info[0]}")
        logging.warning("vs_temporalfix: Engine building complete.")
        return engine_path
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _tensorrt_inference(input_clips, model_files, onnx_dir, engine_dir, strength, clip_w, clip_h, tiles=1, overlap=64, num_streams=1, force_rebuild=False):
    tile_w, tile_h, _, _ = get_tiles(clip_w=clip_w, clip_h=clip_h, tiles=tiles, overlap=overlap)
    engine_path = _get_engine(model_files=model_files, onnx_dir=onnx_dir, engine_dir=engine_dir, strength=strength, engine_w=tile_w, engine_h=tile_h, force_rebuild=force_rebuild)
    model_args  = dict(engine_path=engine_path, num_streams=num_streams, **(dict(tilesize=(tile_w, tile_h), overlap=(overlap, overlap)) if tiles > 1 else {}))

    # try inference, rebuild engine if it fails
    try:
        out = core.trt.Model(input_clips, **model_args)
    except vs.Error as e:
        err_msg = str(e).lower()
        serialization_keywords = ("serialize", "serialization", "deserialize", "deserialization")
        if any(k in err_msg for k in serialization_keywords) and not force_rebuild:
            logging.warning("vs_temporalfix: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
            model_args["engine_path"] = _get_engine(model_files=model_files, onnx_dir=onnx_dir, engine_dir=engine_dir, strength=strength, engine_w=tile_w, engine_h=tile_h, force_rebuild=True)
            out = core.trt.Model(input_clips, **model_args)
        else:
            raise
    return out


def _tensorrt(clip, strength=2, tiles=1, num_streams=1, engine_folder=None, exclude=None):
    
    # checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_temporalfix: Clip must be a vapoursynth clip.")
    if clip.format.id  == vs.PresetVideoFormat.NONE or clip.width  == 0 or clip.height  == 0:
        raise TypeError("vs_temporalfix: Clip must have constant format and dimensions.")
    if clip.num_frames < 4:
        raise ValueError("vs_temporalfix: Clip must be at least 4 frames long.")
    if clip.format.id not in [vs.RGBH]:
        raise ValueError("vs_temporalfix: Clip must be in RGBH format for the TensorRT backend.")
    if strength < 0 or strength > 3:
        raise ValueError("vs_temporalfix: Strength must be in the 0.0-3.0 range.")
    if num_streams < 1:
        raise ValueError("vs_temporalfix: Number of parallel TensorRT streams (num_streams) must be at least 1.")
    
    # select model
    orig_clip     = clip
    strength      = round(float(strength), 2)
    clip_w        = clip.width
    clip_h        = clip.height
    overlap       = min(64, int(64 * ((clip_w * clip_h) / (1920 * 1080)) ** 0.5))  # overlap gets smaller for smaller inputs
    force_rebuild = False
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    onnx_dir      = os.path.join(current_dir, "models")
    engine_dir    = os.path.join(current_dir, "engines") if engine_folder is None else os.path.abspath(engine_folder)
    model_files   = {
        0: "temporalfix_s0_v1_op18_fp16.onnx",
        1: "temporalfix_s1_v1.1_op18_fp16.onnx",
        2: "temporalfix_s2_v1_op18_fp16.onnx",
        3: "temporalfix_s3_v1_op18_fp16.onnx",
    }

    if strength == 0:
        return clip

    # shift and inference
    clip = basic_expr(clip, expr=["x 0 max 1 min"])  # clamp
    input_clips = gen_shifts(clip, radius=3)  # [-3, -2, -1, 0, +1, +2, +3]
    out = _tensorrt_inference(input_clips=input_clips, model_files=model_files, onnx_dir=onnx_dir, engine_dir=engine_dir, strength=strength, clip_w=clip_w, clip_h=clip_h, tiles=tiles, overlap=overlap, num_streams=num_streams, force_rebuild=force_rebuild)
    out = core.std.CopyFrameProps(out, clip)  # copy props to make sure they are not from the shifted -3 clip
    return exclude_regions(out, orig_clip, exclude=exclude)  # exclude regions from temporalfix


def model(clip, strength=2, tiles=1, backend="tensorrt", num_streams=1, engine_folder=None, exclude=None):
    """Add temporal coherence to single image AI upscaling models. Also known as temporal consistency, line wiggle fix, stabilization, deshimmering.

    Args:
        clip: Temporally unstable upscaled clip.
        strength: Suppression strength of temporal inconsistencies in the range `0.0-3.0`. Higher means more aggressive. 
            Higher resolution tends to need higher strength. Too high may oversmooth small movements.
        tiles: A higher amount of tiles will reduce VRAM usage at the cost of speed. 
            This should only be needed on low end hardware. `tiles=1` will use the full frame, which is fastest.
        backend: The backend used to run the model.
            - `cpu` = CPU mode using PyTorch (very slow).
            - `cuda` = GPU mode using PyTorch with CUDA support. Requires any Nvidia GPU (fast).
            - `tensorrt` = GPU mode using vs-mlrt with TensorRT support. Requires an Nvidia RTX GPU (very fast).
        num_streams: Number of parallel TensorRT streams. For high end GPUs higher can be faster, but requires more VRAM. Only affects the TensorRT backend.
        engine_folder: Optional path to the TensorRT engine storage location. By default engines are stored in `vs_temporalfix/engines`. Only affects the TensorRT backend.
        exclude: Optionally exclude scenes with intended temporal inconsistencies. Brackets define excluded frame ranges. 
            Example for two scenes: `exclude="[10 20] [600 900]"`
    """
    
    if backend in ["cpu", "cuda"]:
        return _pytorch(clip, strength=strength, tiles=tiles, device=backend, exclude=exclude)
    if backend in ["tensorrt", "trt"]:
        return _tensorrt(clip, strength=strength, tiles=tiles, num_streams=num_streams, engine_folder=engine_folder, exclude=exclude)
    raise ValueError("vs_temporalfix: Backend must be 'cpu', 'cuda', or 'tensorrt'.")
