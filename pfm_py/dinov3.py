###############################################################################
# DINOv3 - Shape Feature Extraction (compatible with your current pipeline)
###############################################################################

import time
import random
import torch
import numpy as np
from PIL import Image

from torchvision import transforms as tfs
import torchvision
import os
from transformers import AutoImageProcessor, AutoModel


try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import Textures
    from pytorch3d.renderer.cameras import (
        look_at_view_transform,
        PerspectiveCameras,
    )
    from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
    from pytorch3d.renderer.mesh.shader import HardPhongShader
    from pytorch3d.renderer import MeshRenderer
    from pytorch3d.renderer.lighting import PointLights
    PYTORCH3D_AVAILABLE = True
except Exception:
    PYTORCH3D_AVAILABLE = False


# -------------------- GLOBAL SETTINGS --------------------

_DINO_MODEL = None
_DINO_PROCESSOR = None

# DINOv3 ViT-B/16 feature dim (confirmed: 768)
FEATURE_DIMS = 768

PATCH_SIZE = 16     # DINOv3 patch size


# -------------------- MODEL INIT --------------------

def init_dino(device):
    """
    Initialize DINOv3 model using Hugging Face Transformers to avoid fbaipublicfiles.
    Returns processor + model.
    """

    global _DINO_MODEL, _DINO_PROCESSOR

    pretrained_model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    print(f"[INFO] Loading DINOv3 from Hugging Face: {pretrained_model_name}")
    hf_token = os.environ.get("HF_TOKEN")
    processor = None
    try:
        processor = AutoImageProcessor.from_pretrained(pretrained_model_name, token=hf_token)
    except Exception as e:
        print(f"[WARN] Failed to load HF image processor ({e}). Falling back to torchvision transforms.")
        processor = None

    # trust_remote_code in case model class is custom; supply token if set
    model = AutoModel.from_pretrained(pretrained_model_name, trust_remote_code=True, token=hf_token)
    model = model.to(device).eval()

    _DINO_MODEL = model
    _DINO_PROCESSOR = processor
    return processor, model


def get_dino_model(device):
    global _DINO_MODEL, _DINO_PROCESSOR
    if _DINO_MODEL is None:
        return init_dino(device)
    return _DINO_PROCESSOR, _DINO_MODEL


# -------------------- DINOv3 Dense Feature Extraction --------------------

@torch.no_grad()
def get_dino_features(device, processor, model, img_pil, grid):
    """
    Extract dense DINOv3 patch-level features and map to 256x256 grid.
    Equivalent to your previous get_dino_features() for DINOv2.
    """

    # 1) Use HF processor when available; otherwise fall back to ImageNet transforms
    if processor is not None:
        inputs = processor(images=img_pil, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
    else:
        transform = tfs.Compose([
            tfs.Resize(518, interpolation=tfs.InterpolationMode.BICUBIC),
            tfs.CenterCrop(518),
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(img_pil).unsqueeze(0).to(device)

    # 2) Forward pass
    # 2) Forward pass via HF model
    outputs = model(pixel_values=pixel_values)
    tokens = outputs.last_hidden_state  # [B, N_tokens, C]
    # tokens: [CLS, REG1..REG4, PATCHES]
    patch_tokens = tokens[:, 5:, :]

    B, N, C = patch_tokens.shape

    # 3) Convert patch tokens to spatial grid
    side = int(np.sqrt(N))     # Should be sqrt(#patches)
    patch_tokens = patch_tokens.reshape(B, side, side, C)
    patch_tokens = patch_tokens.permute(0, 3, 1, 2)   # [1, C, H_p, W_p]

    # 4) grid_sample to target grid (same as your DINOv2 code)
    features = torch.nn.functional.grid_sample(
        patch_tokens, grid, align_corners=False
    ).reshape(1, C, -1)

    # 5) normalize
    features = torch.nn.functional.normalize(features, dim=1)

    return features



# -------------------- Rendering + Aggregation --------------------

def arange_pixels(resolution=(128, 128), batch_size=1, invert_y_axis=False, device="cuda"):
    h, w = resolution
    uh = 1
    uw = 1
    x = torch.linspace(-uw, uw, w, device=device)
    y = torch.linspace(-uh, uh, h, device=device)
    x, y = torch.meshgrid(x, y)
    pixel_scaled = (
        torch.stack([x, y], -1)
        .permute(1, 0, 2)
        .reshape(1, -1, 2)
        .repeat(batch_size, 1, 1)
    )
    if invert_y_axis:
        pixel_scaled[..., -1] *= -1.0
    return pixel_scaled


def batch_render(device, mesh, mesh_vertices, num_views, H, W):
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0

    distance = torch.sqrt((bb_diff * bb_diff).sum()) * 0.65

    # View grid
    steps = int(np.ceil(np.sqrt(max(4, num_views))))
    end = 360 - 360 / steps
    elev_grid = torch.linspace(0, end, steps)
    azim_grid = torch.linspace(0, end, steps)

    elevation = elev_grid.repeat_interleave(steps)[:num_views]
    azimuth = azim_grid.repeat(steps)[:num_views]

    R, T = look_at_view_transform(
        dist=distance,
        azim=azimuth,
        elev=elevation,
        device=device,
        at=bbox_center.unsqueeze(0)
    )

    camera = PerspectiveCameras(R=R, T=T, device=device)

    rasterization_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1
    )

    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    lights = PointLights(device=device)
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    batch_mesh = mesh.extend(num_views)
    images = renderer(batch_mesh)
    frags = rasterizer(batch_mesh)

    return images, camera, frags.zbuf



# -------------------- DINOv3 per-vertex aggregation --------------------

def get_features_per_vertex(device, processor, model, mesh, mesh_vertices,
                            num_views=32, H=256, W=256):

    device = torch.device(device)
    mesh = mesh.to(device)
    mesh_vertices = mesh_vertices.to(device)

    batched_img, camera, depth = batch_render(device, mesh, mesh_vertices,
                                              num_views, H, W)

    grid = arange_pixels((H, W), invert_y_axis=False, device=device)[0].reshape(1, H, W, 2).to(device)

    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS), device=device)
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1), device=device)


    for i in range(num_views):

        img_np = (batched_img[i, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np)

        dense_feat = get_dino_features(
            device=device,
            processor=processor,
            model=model,
            img_pil=img_pil,
            grid=grid,
        )   # [1, C, H*W]

        dense_feat = dense_feat[0].T     # [H*W, C]

        # naive aggregation (same as your current v2 code)
        ft_per_vertex += dense_feat[:len(mesh_vertices), :]
        ft_per_vertex_count += 1


    ft_per_vertex = ft_per_vertex / ft_per_vertex_count.clamp(min=1)
    return ft_per_vertex



# -------------------- Public API --------------------

def compute_shape_dino_features(verts, faces):
    if not PYTORCH3D_AVAILABLE:
        raise RuntimeError('pytorch3d not available in environment')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processor, dino_model = get_dino_model(device)

    verts = verts.clone().detach().float()
    faces = faces.clone().detach().long()

    verts_rgb = torch.ones_like(verts)[None] * 0.8
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(device)

    features = get_features_per_vertex(
        device=device,
        processor=processor,
        model=dino_model,
        mesh=mesh,
        mesh_vertices=mesh.verts_list()[0],
        num_views=32,
        H=256, W=256,
    )

    return features.cpu()


def get_shape_dino_features(verts, faces, cache_dir=None):
    return compute_shape_dino_features(verts, faces)


# --- Compatibility wrappers expected by the rest of the pipeline ---
def get_dinov3_model(device):
    # returns processor, model
    return get_dino_model(device)


def get_shape_dinov3_features(verts, faces, cache_dir=None):
    return get_shape_dino_features(verts, faces, cache_dir)
