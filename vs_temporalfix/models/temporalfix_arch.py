
# Architecture by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKWARP_GRID_CACHE: dict[tuple[int, int, str, int | None, torch.dtype], torch.Tensor] = {}
_LOWPASS_WEIGHT_CACHE: dict[tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def _make_base_grid(
    h: int,
    w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    horizontal = ((torch.arange(w, device=device, dtype=dtype) + 0.5) * (2.0 / w) - 1.0)
    vertical = ((torch.arange(h, device=device, dtype=dtype) + 0.5) * (2.0 / h) - 1.0)

    horizontal = horizontal.view(1, 1, 1, w).expand(1, -1, h, -1)
    vertical = vertical.view(1, 1, h, 1).expand(1, -1, -1, w)

    return torch.cat((horizontal, vertical), dim=1)


def _make_lowpass_weight_5x5(
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:

    k1 = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], device=device, dtype=dtype)
    k2 = torch.outer(k1, k1)
    k2 = (k2 / k2.sum()).view(1, 1, 5, 5)
    return k2.repeat(channels, 1, 1, 1).contiguous()


def _get_lowpass_weight_5x5(
    channels: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (int(channels), device.type, device.index, dtype)
    weight = _LOWPASS_WEIGHT_CACHE.get(key)
    if weight is None:
        weight = _make_lowpass_weight_5x5(
            int(channels),
            device=device,
            dtype=dtype,
        )
        _LOWPASS_WEIGHT_CACHE[key] = weight
    return weight


def _needs_downsample_prefilter(
    in_hw: tuple[int, int],
    out_hw: tuple[int, int],
) -> bool:
    return out_hw[0] < in_hw[0] or out_hw[1] < in_hw[1]


def _lowpass_before_downsample(
    x: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:

    if not _needs_downsample_prefilter(x.shape[-2:], size):
        return x

    channels = int(x.shape[1])
    weight = _get_lowpass_weight_5x5(
        channels,
        device=x.device,
        dtype=x.dtype,
    )
    return F.conv2d(x, weight, stride=1, padding=2, groups=channels)


def _flow_to_grid_from_base(
    ten_flow: torch.Tensor,
    base_grid: torch.Tensor,
    *,
    fp32_grid: bool,
) -> torch.Tensor:

    flow = ten_flow.float() if fp32_grid else ten_flow
    _, _, h, w = flow.shape

    grid_dtype = torch.float32 if fp32_grid else ten_flow.dtype
    base_grid = base_grid.to(device=flow.device, dtype=grid_dtype)

    flow_x = flow[:, 0:1] * (2.0 / float(w))
    flow_y = flow[:, 1:2] * (2.0 / float(h))
    grid = base_grid + torch.cat((flow_x, flow_y), dim=1)
    return grid.permute(0, 2, 3, 1)


def _flow_to_grid_cached(
    ten_flow: torch.Tensor,
    *,
    fp32_grid: bool,
) -> torch.Tensor:

    flow = ten_flow.float() if fp32_grid else ten_flow
    _, _, h, w = flow.shape
    grid_dtype = torch.float32 if fp32_grid else ten_flow.dtype

    key = (h, w, flow.device.type, flow.device.index, grid_dtype)

    base = _BACKWARP_GRID_CACHE.get(key)
    if base is None:
        base = _make_base_grid(h, w, device=flow.device, dtype=grid_dtype)
        _BACKWARP_GRID_CACHE[key] = base

    flow_x = flow[:, 0:1] * (2.0 / float(w))
    flow_y = flow[:, 1:2] * (2.0 / float(h))
    grid = base + torch.cat((flow_x, flow_y), dim=1)
    return grid.permute(0, 2, 3, 1)


def warp(
    ten_input: torch.Tensor,
    ten_flow: torch.Tensor,
    base_grid: torch.Tensor | None = None,
    *,
    force_fp32: bool = False,
) -> torch.Tensor:

    if base_grid is not None:
        grid = _flow_to_grid_from_base(ten_flow, base_grid, fp32_grid=force_fp32)
    else:
        grid = _flow_to_grid_cached(ten_flow, fp32_grid=force_fp32)

    if force_fp32:
        out = F.grid_sample(
            input=ten_input.float(),
            grid=grid.float(),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return out.to(dtype=ten_input.dtype)

    return F.grid_sample(
        input=ten_input,
        grid=grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def _scaled_hw(hw: tuple[int, int], scale: int) -> tuple[int, int]:
    if scale <= 1:
        return hw
    h, w = hw
    return ((h + scale - 1) // scale, (w + scale - 1) // scale)


def _resize_like(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    x = _lowpass_before_downsample(x, size)
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def _resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if flow.shape[-2:] == size:
        return flow

    old_h, old_w = flow.shape[-2:]
    new_h, new_w = size

    flow = _lowpass_before_downsample(flow, size)
    flow = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    flow_x = flow[:, 0:1] * (float(new_w) / float(old_w))
    flow_y = flow[:, 1:2] * (float(new_h) / float(old_h))
    return torch.cat((flow_x, flow_y), dim=1)


def _expand_scalar_map(
    x: torch.Tensor,
    size: tuple[int, int],
) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    return x[:, :, :1, :1].expand(-1, -1, size[0], size[1])


def conv(
    in_planes: int,
    out_planes: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class Head(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.proj = nn.Conv2d(16, 4, 3, 1, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_hw = x.shape[-2:]
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        x = self.proj(x)
        x = F.interpolate(x, size=input_hw, mode="bilinear", align_corners=False)
        return x


class ResConv(nn.Module):
    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class DSResConv(nn.Module):
    def __init__(self, c: int, dilation: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            c,
            c,
            3,
            1,
            dilation,
            dilation=dilation,
            groups=c,
            bias=True,
        )
        self.pw = nn.Conv2d(c, c, 1, 1, 0, bias=True)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pw(self.dw(x))
        return self.relu(y * self.beta + x)


class MCBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        c: int = 64,
        num_res: int = 4,
        resconv_cls: type[nn.Module] = ResConv,
    ) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(*[resconv_cls(c) for _ in range(num_res)])
        self.out_conv = nn.Conv2d(c, 11, 3, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stage_hw = x.shape[-2:]

        if flow is not None:
            if flow.shape[-2:] != stage_hw:
                raise ValueError(
                    f"MCBlock flow size mismatch: x={stage_hw}, flow={flow.shape[-2:]}"
                )
            x = torch.cat((x, flow), dim=1)

        feat = self.conv0(x)
        feat = self.convblock(feat)

        tmp = self.out_conv(feat)
        tmp = F.interpolate(tmp, size=stage_hw, mode="bilinear", align_corners=False)

        delta_flow = tmp[:, :2]
        conf_logit = tmp[:, 2:3]
        latent = tmp[:, 3:]
        return delta_flow, conf_logit, latent


class PairwiseCenterAligner(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int] = (96, 64, 48),
        num_res: int = 4,
        scale_list: tuple[int, int, int] = (8, 4, 2),
        depthwise_stages: tuple[bool, bool, bool] = (False, True, True),
        fixed_hw: tuple[int, int] | None = None,
        extra_grid_hws: tuple[tuple[int, int], ...] = (),
    ) -> None:
        super().__init__()

        self.scale_list = tuple(int(s) for s in scale_list)
        self.fixed_hw = fixed_hw

        blocks: list[MCBlock] = []
        for idx, (c, use_depthwise) in enumerate(zip(channels, depthwise_stages)):
            resconv_cls = DSResConv if use_depthwise else ResConv
            in_planes = 15 if idx == 0 else 26
            blocks.append(MCBlock(in_planes, c=c, num_res=num_res, resconv_cls=resconv_cls))
        self.blocks = nn.ModuleList(blocks)

        self._static_grid_names: dict[tuple[int, int], str] = {}
        if fixed_hw is not None:
            unique_hws = {tuple(fixed_hw)}
            for s in self.scale_list:
                unique_hws.add(_scaled_hw(fixed_hw, s))
            for hw in extra_grid_hws:
                unique_hws.add(tuple(hw))

            for hw in sorted(unique_hws):
                name = f"_base_grid_{hw[0]}x{hw[1]}"
                self.register_buffer(
                    name,
                    _make_base_grid(hw[0], hw[1], device=torch.device("cpu"), dtype=torch.float32),
                    persistent=False,
                )
                self._static_grid_names[tuple(hw)] = name

    def get_static_base_grid(
        self,
        hw: tuple[int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        del dtype
        name = self._static_grid_names.get(tuple(hw))
        if name is None:
            return None
        return getattr(self, name).to(device=device)

    def _build_center_pyramids(
        self,
        center: torch.Tensor,
        f_center: torch.Tensor,
    ) -> tuple[list[tuple[int, int]], list[torch.Tensor], list[torch.Tensor]]:
        full_hw = center.shape[-2:]
        stage_hws = [_scaled_hw(full_hw, s) for s in self.scale_list]
        center_pyr = [_resize_like(center, hw) for hw in stage_hws]
        f_center_pyr = [_resize_like(f_center, hw) for hw in stage_hws]
        return stage_hws, center_pyr, f_center_pyr

    def _build_support_pyramids(
        self,
        support: torch.Tensor,
        f_support: torch.Tensor,
        delta_t: torch.Tensor,
        stage_hws: list[tuple[int, int]],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        b, s, _, h, w = support.shape

        support_flat = support.reshape(b * s, 3, h, w)
        f_support_flat = f_support.reshape(b * s, 4, h, w)
        delta_t_flat = delta_t.reshape(b * s, 1, h, w)

        support_pyr: list[torch.Tensor] = []
        f_support_pyr: list[torch.Tensor] = []
        delta_t_pyr: list[torch.Tensor] = []
        support_feat_pyr: list[torch.Tensor] = []

        for hw_i in stage_hws:
            support_i = _resize_like(support_flat, hw_i).reshape(b, s, 3, hw_i[0], hw_i[1])
            feat_i = _resize_like(f_support_flat, hw_i).reshape(b, s, 4, hw_i[0], hw_i[1])
            delta_i = _expand_scalar_map(delta_t_flat, hw_i).reshape(b, s, 1, hw_i[0], hw_i[1])

            support_pyr.append(support_i)
            f_support_pyr.append(feat_i)
            delta_t_pyr.append(delta_i)
            support_feat_pyr.append(
                torch.cat((support_i, feat_i), dim=2).reshape(b * s, 7, hw_i[0], hw_i[1])
            )

        return support_pyr, f_support_pyr, delta_t_pyr, support_feat_pyr

    def forward(
        self,
        center: torch.Tensor,
        support: torch.Tensor,
        f_center: torch.Tensor,
        f_support: torch.Tensor,
        delta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _, full_h, full_w = support.shape
        full_hw = (full_h, full_w)

        stage_hws, center_pyr, f_center_pyr = self._build_center_pyramids(center, f_center)
        support_pyr, f_support_pyr, delta_t_pyr, support_feat_pyr = self._build_support_pyramids(
            support=support,
            f_support=f_support,
            delta_t=delta_t,
            stage_hws=stage_hws,
        )

        support_flat_full = support.reshape(b * s, 3, full_h, full_w)
        ones_full = support_flat_full.new_ones((b * s, 1, full_h, full_w))

        flow = None
        conf_logit = None
        latent = None

        for stage_idx, block in enumerate(self.blocks):
            stage_hw = stage_hws[stage_idx]

            center_i = (
                center_pyr[stage_idx][:, None]
                .expand(-1, s, -1, -1, -1)
                .reshape(b * s, 3, stage_hw[0], stage_hw[1])
            )
            f_center_i = (
                f_center_pyr[stage_idx][:, None]
                .expand(-1, s, -1, -1, -1)
                .reshape(b * s, 4, stage_hw[0], stage_hw[1])
            )
            support_i = support_pyr[stage_idx].reshape(b * s, 3, stage_hw[0], stage_hw[1])
            f_support_i = f_support_pyr[stage_idx].reshape(b * s, 4, stage_hw[0], stage_hw[1])
            delta_t_i = delta_t_pyr[stage_idx].reshape(b * s, 1, stage_hw[0], stage_hw[1])

            if flow is not None and flow.shape[-2:] != stage_hw:
                flow = _resize_flow(flow, stage_hw)
                conf_logit = _resize_like(conf_logit, stage_hw)
                latent = _resize_like(latent, stage_hw)

            if flow is None:
                x = torch.cat((center_i, support_i, f_center_i, f_support_i, delta_t_i), dim=1)
                delta_flow, conf_logit, latent = block(x, None)
                flow = delta_flow
            else:
                base_grid = self.get_static_base_grid(
                    stage_hw,
                    device=flow.device,
                    dtype=flow.dtype,
                )
                warped_cat = warp(
                    support_feat_pyr[stage_idx],
                    flow,
                    base_grid=base_grid,
                    force_fp32=False,
                )
                warped = warped_cat[:, :3]
                warped_feat = warped_cat[:, 3:7]

                x = torch.cat(
                    (
                        center_i,
                        warped,
                        f_center_i,
                        warped_feat,
                        delta_t_i,
                        conf_logit,
                        latent,
                    ),
                    dim=1,
                )
                delta_flow, conf_logit, latent = block(x, flow)
                flow = flow + delta_flow

        if flow is None or conf_logit is None:
            raise RuntimeError("Internal error: flow/conf_logit was not initialized")

        if flow.shape[-2:] != full_hw:
            flow = _resize_flow(flow, full_hw)
            conf_logit = _resize_like(conf_logit, full_hw)

        base_grid_full = self.get_static_base_grid(
            full_hw,
            device=flow.device,
            dtype=flow.dtype,
        )

        warped = warp(
            support_flat_full,
            flow,
            base_grid=base_grid_full,
            force_fp32=True,
        )
        valid = warp(
            ones_full,
            flow,
            base_grid=base_grid_full,
            force_fp32=False,
        )

        return warped, valid, conf_logit, flow


class ConfidenceHead(nn.Module):
    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            conv(25, hidden, 3, 1, 1),
            conv(hidden, hidden, 3, 1, 1),
            nn.Conv2d(hidden, 1, 3, 1, 1),
        )

    def forward(
        self,
        center: torch.Tensor,
        warped: torch.Tensor,
        f_center: torch.Tensor,
        f_warped: torch.Tensor,
        flow: torch.Tensor,
        delta_t: torch.Tensor,
        raw_conf_logit: torch.Tensor,
    ) -> torch.Tensor:
        rgb_abs = torch.abs(warped - center)
        feat_abs = torch.abs(f_warped - f_center)
        flow_mag = torch.linalg.vector_norm(flow, ord=2, dim=1, keepdim=True)

        x = torch.cat(
            (
                center,
                warped,
                rgb_abs,
                f_center,
                f_warped,
                feat_abs,
                flow,
                flow_mag,
                delta_t,
            ),
            dim=1,
        )
        return raw_conf_logit + self.net(x)


def conservative_temporal_average(
    center: torch.Tensor,
    aligned_supports: torch.Tensor,
    support_conf: torch.Tensor,
    conf_thresh: float = 0.6,
    min_support: int = 1,
    gate_slope: float = 12.0,
    count_slope: float = 4.0,
) -> torch.Tensor:
    if aligned_supports.ndim != 5:
        raise ValueError(
            f"aligned_supports must be [B, S, 3, H, W], got {aligned_supports.shape}"
        )
    if support_conf.ndim != 5:
        raise ValueError(f"support_conf must be [B, S, 1, H, W], got {support_conf.shape}")
    if center.ndim != 4:
        raise ValueError(f"center must be [B, 3, H, W], got {center.shape}")

    gate = torch.sigmoid((support_conf - conf_thresh) * gate_slope)
    weights = support_conf * gate

    weighted_sum = (weights * aligned_supports).sum(dim=1)
    weight_sum = weights.sum(dim=1)
    soft_count = gate.sum(dim=1)

    blended = (center + weighted_sum) / (1.0 + weight_sum).clamp_min(1e-6)

    if min_support > 0:
        readiness = torch.sigmoid((soft_count - float(min_support) + 0.5) * count_slope)
        averaged = center + readiness * (blended - center)
    else:
        averaged = blended

    return averaged


class temporalfix_arch(nn.Module):
    def __init__(
        self,
        channels: tuple[int, int, int] = (96, 64, 48),
        num_res: int = 4,
        agreement_scale: float | None = None,
        scale_list: tuple[int, int, int] = (8, 4, 2),
        conf_thresh: float = 0.6,
        min_support: int = 1,
        gate_slope: float = 12.0,
        count_slope: float = 4.0,
        scale: int = 1,
        fixed_hw: tuple[int, int] | None = None,
        depthwise_stages: tuple[bool, bool, bool] = (False, True, True),
        confidence_scale: int = 2,
    ) -> None:
        super().__init__()

        self.scale = scale
        self.scale_list = tuple(int(s) for s in scale_list)
        self.conf_thresh = float(conf_thresh)
        self.min_support = int(min_support)
        self.gate_slope = float(gate_slope)
        self.count_slope = float(count_slope)
        self.agreement_scale = agreement_scale
        self.fixed_hw = fixed_hw
        self.confidence_scale = int(confidence_scale)

        self.register_buffer(
            "_temporal_values",
            torch.tensor(
                [-1.0, -2.0 / 3.0, -1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
                dtype=torch.float32,
            ).view(1, 6, 1, 1, 1),
            persistent=False,
        )

        extra_grid_hws: tuple[tuple[int, int], ...] = ()
        if fixed_hw is not None:
            extra_grid_hws = (_scaled_hw(fixed_hw, self.confidence_scale),)

        self.encode = Head()
        self.aligner = PairwiseCenterAligner(
            channels=channels,
            num_res=num_res,
            scale_list=self.scale_list,
            depthwise_stages=depthwise_stages,
            fixed_hw=fixed_hw,
            extra_grid_hws=extra_grid_hws,
        )
        self.conf_head = ConfidenceHead(hidden=32)

    def _reshape_input(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, int, int, int]:
        if x.ndim != 5:
            raise ValueError(f"Expected Bx7x3xHxW input, got {x.shape}")

        frames = x
        b, t, c, h, w = frames.shape
        if t != 7 or c != 3:
            raise ValueError(f"Expected Bx7x3xHxW, got {frames.shape}")

        return frames, b, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames, b, h, w = self._reshape_input(x)

        if self.fixed_hw is not None and (h, w) != tuple(self.fixed_hw):
            raise ValueError(
                f"Input size {(h, w)} does not match fixed_hw={self.fixed_hw}"
            )

        center = frames[:, 3]

        feats = self.encode(frames.reshape(b * 7, 3, h, w)).reshape(b, 7, 4, h, w)
        center_feat = feats[:, 3]

        frame_indices = (0, 1, 2, 4, 5, 6)
        supports = frames[:, frame_indices]
        support_feats = feats[:, frame_indices]
        num_supports = supports.shape[1]

        delta_t = self._temporal_values.to(device=center.device, dtype=center.dtype).expand(
            b, -1, -1, h, w
        )

        warped, warped_valid, raw_conf, final_flow = self.aligner(
            center=center,
            support=supports,
            f_center=center_feat,
            f_support=support_feats,
            delta_t=delta_t,
        )

        conf_hw = _scaled_hw((h, w), self.confidence_scale)

        center_half = _resize_like(center, conf_hw)
        center_feat_half = _resize_like(center_feat, conf_hw)
        raw_conf_half = _resize_like(raw_conf, conf_hw)
        delta_t_half = _expand_scalar_map(delta_t.reshape(b * num_supports, 1, h, w), conf_hw)
        flow_half = _resize_flow(final_flow, conf_hw)

        warped_half = _resize_like(warped, conf_hw)
        support_feat_half = _resize_like(
            support_feats.reshape(b * num_supports, 4, h, w),
            conf_hw,
        )

        base_grid_half = self.aligner.get_static_base_grid(
            conf_hw,
            device=flow_half.device,
            dtype=flow_half.dtype,
        )
        warped_feat_half = warp(
            support_feat_half,
            flow_half,
            base_grid=base_grid_half,
            force_fp32=False,
        )

        center_rep_half = (
            center_half[:, None]
            .expand(-1, num_supports, -1, -1, -1)
            .reshape(b * num_supports, 3, conf_hw[0], conf_hw[1])
        )
        center_feat_rep_half = (
            center_feat_half[:, None]
            .expand(-1, num_supports, -1, -1, -1)
            .reshape(b * num_supports, 4, conf_hw[0], conf_hw[1])
        )

        conf_logit_half = self.conf_head(
            center=center_rep_half,
            warped=warped_half,
            f_center=center_feat_rep_half,
            f_warped=warped_feat_half,
            flow=flow_half,
            delta_t=delta_t_half,
            raw_conf_logit=raw_conf_half,
        )
        conf_logit = _resize_like(conf_logit_half, (h, w))

        conf = torch.sigmoid(conf_logit) * warped_valid

        aligned_supports = warped.reshape(b, num_supports, 3, h, w)
        support_conf = conf.reshape(b, num_supports, 1, h, w)

        averaged = conservative_temporal_average(
            center=center,
            aligned_supports=aligned_supports,
            support_conf=support_conf,
            conf_thresh=self.conf_thresh,
            min_support=self.min_support,
            gate_slope=self.gate_slope,
            count_slope=self.count_slope,
        )

        return averaged
