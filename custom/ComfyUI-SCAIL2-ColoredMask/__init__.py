# SCAIL2ColoredMask — single-node back-port from ComfyUI master (comfy_extras/nodes_scail.py)
# into stable v0.20.3, which already ships everything this node needs (SAM3 backend,
# WanSCAILToVideo, comfy.ldm.sam3.tracker.unpack_masks, comfy_api.latest io). This adds ONLY
# the missing SCAIL2ColoredMask node so the SCAIL-2 GGUF ComfyUI workflow loads with no red node
# and WITHOUT updating the ComfyUI core. Source: github.com/comfyanonymous/ComfyUI master.
from typing_extensions import override

import torch
import torch.nn.functional as F

import comfy.model_management
from comfy_api.latest import ComfyExtension, io
from comfy.ldm.sam3.tracker import unpack_masks

SAM3TrackData = io.Custom("SAM3_TRACK_DATA")

# Model was trained on these exact colors; deviating degrades multi-identity quality.
DEFAULT_PALETTE = [
    (0.0, 0.0, 1.0),  # Blue
    (1.0, 0.0, 0.0),  # Red
    (0.0, 1.0, 0.0),  # Green
    (1.0, 0.0, 1.0),  # Magenta
    (0.0, 1.0, 1.0),  # Cyan
    (1.0, 1.0, 0.0),  # Yellow
]


def _unpack(track_data):
    packed = track_data["packed_masks"]
    if packed is None or packed.shape[1] == 0:
        return None
    return unpack_masks(packed)


def _first_appearance_cx_area(masks_bool):
    """Per object: first frame it appears in, plus centroid-x and area in that frame."""
    m = masks_bool.float()
    T, H, W = m.shape[0], m.shape[-2], m.shape[-1]
    grid_x = torch.arange(W, device=m.device, dtype=m.dtype).view(1, 1, 1, W)
    area_t = m.sum(dim=(-1, -2))
    cx_t = (m * grid_x).sum(dim=(-1, -2)) / area_t.clamp(min=1)
    present = area_t > 0
    frame_idx = torch.arange(T, device=m.device).unsqueeze(1)
    first_t = torch.where(present, frame_idx, T).amin(dim=0)
    sel = first_t.clamp(max=T - 1).unsqueeze(0)
    cx = cx_t.gather(0, sel).squeeze(0)
    area = area_t.gather(0, sel).squeeze(0)
    return first_t.tolist(), (cx / W).tolist(), (area / (H * W)).tolist()


def _subset_track_data(track_data, obj_indices):
    out = dict(track_data)
    packed = track_data["packed_masks"]
    if packed is None or not obj_indices:
        out["packed_masks"] = None
        if "scores" in out:
            out["scores"] = []
        return out
    out["packed_masks"] = packed[:, obj_indices].contiguous()
    scores = track_data.get("scores")
    if scores is not None:
        out["scores"] = [scores[i] for i in obj_indices if i < len(scores)]
    return out


def _render_colored_masks(track_data, background="black"):
    packed = track_data["packed_masks"]
    H, W = track_data["orig_size"]
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    if packed is None or packed.shape[1] == 0:
        T = track_data.get("n_frames", 1) if packed is None else packed.shape[0]
        out = torch.empty(T, H, W, 3, device=device, dtype=dtype)
        out[..., 0], out[..., 1], out[..., 2] = bg_rgb[0], bg_rgb[1], bg_rgb[2]
        return out
    T, N_obj = packed.shape[0], packed.shape[1]
    colors = torch.tensor(
        [DEFAULT_PALETTE[i % len(DEFAULT_PALETTE)] for i in range(N_obj)],
        device=device, dtype=dtype,
    )
    masks_full = unpack_masks(packed.to(device)).float()
    Hm, Wm = masks_full.shape[-2], masks_full.shape[-1]
    masks_full = F.interpolate(
        masks_full.view(T * N_obj, 1, Hm, Wm), size=(H, W), mode="nearest"
    ).view(T, N_obj, H, W) > 0.5
    any_mask = masks_full.any(dim=1)
    color_overlay = colors[masks_full.to(torch.uint8).argmax(dim=1)]
    bg_tensor = torch.tensor(bg_rgb, device=device, dtype=color_overlay.dtype).view(1, 1, 1, 3)
    return torch.where(any_mask.unsqueeze(-1), color_overlay, bg_tensor.expand_as(color_overlay))


def _render_mask_as_identity(mask, background="black"):
    """Plain comfy MASK (B,H,W) or (H,W) -> (B,H,W,3) rendered as a single identity (palette[0])
    on the given background. A batch is treated as multiple views of that one subject."""
    device = comfy.model_management.intermediate_device()
    dtype = comfy.model_management.intermediate_dtype()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    mask = mask.to(device=device, dtype=dtype)
    B, H, W = mask.shape
    bg_rgb = (1.0, 1.0, 1.0) if background.startswith("white") else (0.0, 0.0, 0.0)
    color = torch.tensor(DEFAULT_PALETTE[0], device=device, dtype=dtype).view(1, 1, 1, 3)
    bg = torch.tensor(bg_rgb, device=device, dtype=dtype).view(1, 1, 1, 3)
    return torch.where((mask > 0.5).unsqueeze(-1), color.expand(B, H, W, 3), bg.expand(B, H, W, 3))


class SCAIL2ColoredMask(io.ComfyNode):
    """Render SAM3 tracks for the driving pose video and reference image(s) into the
    colored masks WanSCAILToVideo consumes. Shared `sort_by` keeps each identity on the
    same color across both outputs.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2ColoredMask",
            display_name="Create SCAIL-2 Colored Mask",
            category="model/conditioning/wan/scail",
            inputs=[
                SAM3TrackData.Input("driving_track_data", tooltip="SAM3 track of the driving pose video. Will be rendered into the pose_video_mask output."),
                io.MultiType.Input("ref_track_data", [SAM3TrackData, io.Mask], optional=True, display_name="reference_masks",
                                   tooltip="SAM3 track of the reference image(s) (one identity per object, colored in batch order), or a plain MASK of the reference subject (rendered as a single identity)."),
                io.String.Input("object_indices", default="",
                                tooltip="Comma-separated list of person indices to include (e.g. '0,2,3'). Applied to both reference and pose video masks. Empty = all."),
                io.Combo.Input("sort_by", options=["none", "left_to_right", "area"], default="left_to_right",
                               tooltip="Order in which palette colors are assigned to the tracked objects (applied to both reference and pose video so each identity keeps the same color). Objects that appear in earlier frames always come first; within a frame, left_to_right = leftmost object (by centroid at first appearance) gets the first color, area = biggest object (by mask area at first appearance) gets the first color; none = keep SAM3's order."),
                io.Boolean.Input("replacement_mode", default=False,
                    tooltip="False = Animation Mode (pose_video_mask has black background, reference_image_mask has white background). "
                    "True = Replacement Mode (pose_video_mask has white background, reference_image_mask has black background)."),
            ],
            outputs=[
                io.Image.Output("pose_video_mask"),
                io.Image.Output("reference_image_mask"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, driving_track_data, object_indices, sort_by, replacement_mode, ref_track_data=None):
        def _prep(td):
            masks_bool = _unpack(td)
            if sort_by != "none" and masks_bool is not None:
                first_t, cx, area = _first_appearance_cx_area(masks_bool)
                if sort_by == "left_to_right":
                    order = sorted(range(len(cx)), key=lambda i: (first_t[i], cx[i]))
                else:  # "area"
                    order = sorted(range(len(area)), key=lambda i: (first_t[i], -area[i]))
                td = _subset_track_data(td, order)
            if object_indices.strip():
                indices = [int(i.strip()) for i in object_indices.split(",") if i.strip().isdigit()]
                packed = td.get("packed_masks")
                n_obj = packed.shape[1] if packed is not None else 0
                indices = [i for i in indices if 0 <= i < n_obj]
                td = _subset_track_data(td, indices)
            return td

        drv = _prep(driving_track_data)
        # Animation: driving=black, ref=white. Replacement: driving=white, ref=black.
        mask_video = _render_colored_masks(drv, "white" if replacement_mode else "black")
        ref_bg = "black" if replacement_mode else "white"

        if ref_track_data is not None:
            if isinstance(ref_track_data, torch.Tensor):  # plain comfy MASK
                reference_image_mask = _render_mask_as_identity(ref_track_data, ref_bg)
            else:
                reference_image_mask = _render_colored_masks(_prep(ref_track_data), ref_bg)
        else:
            H, W = drv["orig_size"]
            fill_value = 1.0 if ref_bg == "white" else 0.0
            reference_image_mask = torch.full((1, H, W, 3), fill_value, device=comfy.model_management.intermediate_device(), dtype=comfy.model_management.intermediate_dtype())

        return io.NodeOutput(mask_video, reference_image_mask)


class SCAIL2ColoredMaskExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SCAIL2ColoredMask]


async def comfy_entrypoint() -> SCAIL2ColoredMaskExtension:
    return SCAIL2ColoredMaskExtension()
