################################################################################
## DINOv3 descriptor computation
## References:
## 1. DINOv3 Model Repository (Facebook Research)
##    https://github.com/facebookresearch/dinov3
##    Provides the DINOv3 vision transformer model and utilities
##
## 2. Echo-Match Repository (Application built on DINOv3) 
##    Code taken from this repository and slightly modified.
##    Source: https://github.com/vikiehm/echo-match/blob/main/utils/dino_util.py
##
################################################################################

from dataclasses import dataclass
import os
import math
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tfs

try:
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import Textures
    from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
    from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
    from pytorch3d.renderer.mesh.shader import HardPhongShader
    from pytorch3d.renderer import MeshRenderer
    from pytorch3d.renderer.lighting import PointLights
    from pytorch3d.ops import ball_query
    PYTORCH3D_AVAILABLE = True
except Exception:
    PYTORCH3D_AVAILABLE = False

from transformers import AutoModel

PATCH_SIZE = 16
NUM_SKIP_TOKENS = 5  # CLS + 4 register tokens
FEATURE_DIMS = 768  # ViT-L
VERTEX_GPU_LIMIT = 35000
_DINO_MODEL = None


def get_dino_model(device):
    global _DINO_MODEL
    if _DINO_MODEL is None:
        model_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
        hf_token = os.environ.get("HF_TOKEN")
        _DINO_MODEL = AutoModel.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        _DINO_MODEL = _DINO_MODEL.to(device).eval()
    return _DINO_MODEL


@torch.no_grad()
def get_dino_features(device, dino_model, img, grid):
    transform = tfs.Compose([
        tfs.Resize((518, 518)),
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)[:3].unsqueeze(0).to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        outputs = dino_model(img)
    patch_tokens = outputs.last_hidden_state[:, NUM_SKIP_TOKENS:, :].half()
    h, w = img.shape[2] // PATCH_SIZE, img.shape[3] // PATCH_SIZE
    features = patch_tokens.reshape(-1, h, w, FEATURE_DIMS).permute(0, 3, 1, 2)
    features = torch.nn.functional.grid_sample(features, grid, align_corners=False)
    features = features.reshape(1, FEATURE_DIMS, -1)
    return torch.nn.functional.normalize(features, dim=1)


def arange_pixels(resolution, invert_y_axis=False, device="cuda"):
    h, w = resolution
    x = torch.linspace(-1, 1, w, device=device)
    y = torch.linspace(-1, 1, h, device=device)
    x, y = torch.meshgrid(x, y, indexing='xy')
    pixels = torch.stack([x, y], -1).reshape(1, -1, 2)
    if invert_y_axis:
        pixels[..., -1] *= -1.0
    return pixels


def batch_render(compute_opts, device, mesh, num_views, H, W):
    steps = int(math.ceil(math.sqrt(num_views)))
    end = 360 - 360 / steps
    elevation = torch.linspace(0, end, steps).repeat(steps)[:num_views]
    azimuth = torch.linspace(0, end, steps).repeat_interleave(steps)[:num_views]

    R, T = look_at_view_transform(dist=compute_opts.cam_distance, azim=azimuth, elev=elevation, device=device, at=compute_opts.cam_target.unsqueeze(0))
    camera = PerspectiveCameras(R=R, T=T, device=device)
    raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=raster_settings)
    lights = PointLights(device=device, location=camera.get_camera_center())
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    batch_mesh = mesh.extend(num_views)
    images = renderer(batch_mesh)
    depth = rasterizer(batch_mesh).zbuf
    return images, camera, depth

@dataclass
class DinoV3ComputeOpts:
    cam_distance: float
    cam_target: torch.Tensor
    global_scale: float

def get_compute_opts(verts) -> DinoV3ComputeOpts:
    bbox_min = verts.min(dim=0).values
    bbox_max = verts.max(dim=0).values
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_diag = torch.norm(bbox_max - bbox_min)
    cam_distance = bbox_diag * 0.65

    return DinoV3ComputeOpts(cam_distance=cam_distance, cam_target=bbox_center, global_scale=bbox_diag)

def compute_dinov3_features(compute_opts: DinoV3ComputeOpts, verts, faces, num_views=100, H=512, W=512, tolerance=0.004):
    if not PYTORCH3D_AVAILABLE:
        raise RuntimeError('pytorch3d not available')
    if compute_opts is None:
        compute_opts = get_compute_opts(verts)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = get_dino_model(device)

    verts = verts.clone().detach().float()
    faces = faces.clone().detach().long()
    textures = Textures(verts_rgb=torch.ones_like(verts)[None] * 0.8)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(device)
    mesh_vertices = mesh.verts_list()[0]

    ball_radius = compute_opts.global_scale * tolerance

    images, camera, depth = batch_render(compute_opts, device, mesh, num_views, H, W)
    pixel_coords = arange_pixels((H, W), invert_y_axis=True, device=device)[0]
    pixel_coords[:, 0] = torch.flip(pixel_coords[:, 0], dims=[0])
    grid = arange_pixels((H, W), invert_y_axis=False, device=device)[0].reshape(1, H, W, 2).half()

    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS), device=device).half()
    ft_count = torch.zeros((len(mesh_vertices), 1), device=device).half()

    for idx in range(num_views):
        dp = depth[idx].flatten().unsqueeze(1)
        xy_depth = torch.cat((pixel_coords, dp), dim=1)
        visible = xy_depth[:, 2] != -1
        xy_depth = xy_depth[visible]

        world_coords = camera[idx].unproject_points(xy_depth, world_coordinates=True, from_ndc=True)
        img = (images[idx, :, :, :3].cpu().numpy() * 255).astype(np.uint8)
        dino_feats = get_dino_features(device, dino_model, Image.fromarray(img), grid)
        feats_visible = dino_feats[0, :, visible]

        queried = ball_query(
            world_coords.unsqueeze(0),
            mesh_vertices.unsqueeze(0),
            K=100,
            radius=ball_radius,
            return_nn=False,
        ).idx[0]

        mask = queried != -1
        repeat_counts = mask.sum(dim=1)
        ft_count[queried[mask]] += 1
        ft_per_vertex[queried[mask]] += feats_visible.repeat_interleave(repeat_counts, dim=1).T

    has_feats = ft_count[:, 0] != 0
    ft_per_vertex[has_feats] /= ft_count[has_feats]

    missing = ~has_feats
    n_missing = missing.sum().item()
    if n_missing > 0:
        print(f"Warning: {n_missing} vertices missing features, using nearest neighbor")
        dists = torch.cdist(mesh_vertices[missing], mesh_vertices[has_feats])
        nearest = dists.argmin(dim=1)
        ft_per_vertex[missing] = ft_per_vertex[has_feats][nearest]

    return ft_per_vertex.cpu(), n_missing
