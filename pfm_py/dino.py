import time
import random
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as tfs

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
    from pytorch3d.ops import ball_query
    PYTORCH3D_AVAILABLE = True
except Exception:
    PYTORCH3D_AVAILABLE = False

# Lightweight DINO integration for pipeline
_DINO_MODEL = None
FEATURE_DIMS = 768
VERTEX_GPU_LIMIT = 35000

def init_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    return model

def get_dino_model(device):
    global _DINO_MODEL
    if _DINO_MODEL is None:
        _DINO_MODEL = init_dino(device)
    return _DINO_MODEL

@torch.no_grad()
def get_dino_features(device, dino_model, img, grid, idx):
    patch_size = 14
    transform = tfs.Compose([
        tfs.Resize((518, 518)),
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)[:3].unsqueeze(0).to(device)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    h = int(img.shape[2] / patch_size)
    w = int(img.shape[3] / patch_size)
    dim = features.shape[-1]
    features = features.reshape(-1, h, w, dim).permute(0, 3, 1, 2)
    features = torch.nn.functional.grid_sample(features, grid, align_corners=False).reshape(1, dim, -1)
    features = torch.nn.functional.normalize(features, dim=1)
    return features


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


def batch_render(device, mesh, mesh_vertices, num_views, H, W, use_normal_map=False):
    # simplified rendering wrapper using PyTorch3D
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]
    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    scaling_factor = 0.65
    distance = torch.sqrt((bb_diff * bb_diff).sum()) * scaling_factor
    # Choose a grid of views with at least num_views samples, then truncate to num_views
    steps = int(np.ceil(np.sqrt(max(4, num_views))))
    end = 360 - 360 / steps
    # create a grid of size (steps*steps) and then take first num_views entries
    elev_grid = torch.linspace(start=0, end=end, steps=steps)
    azim_grid = torch.linspace(start=0, end=end, steps=steps)
    # optional small random offsets (kept zero here)
    add_angle_ele = 0
    add_angle_azi = 0
    elevation = elev_grid.repeat_interleave(steps) + add_angle_ele
    azimuth = azim_grid.repeat(steps) + add_angle_azi
    # Truncate to exactly num_views
    elevation = elevation[:num_views]
    azimuth = azimuth[:num_views]
    rotation, translation = look_at_view_transform(dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center.unsqueeze(0))
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    rasterization_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0)
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    lights = PointLights(device=device)
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    batch_mesh = mesh.extend(num_views)
    batched_renderings = batch_renderer(batch_mesh)
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    return batched_renderings, None, camera, depth


def get_features_per_vertex(device, dino_model, mesh, mesh_vertices, num_views=32, H=256, W=256, tolerance=0.01):
    # Simplified aggregation: render views, extract DINO patches and aggregate per vertex
    device = torch.device(device)
    mesh = mesh.to(device)
    mesh_vertices = mesh_vertices.to(device)
    batched_renderings, _, camera, depth = batch_render(device, mesh, mesh_vertices, num_views, H, W)
    pixel_coords = arange_pixels((H, W), invert_y_axis=True, device=device)[0]
    grid = arange_pixels((H, W), invert_y_axis=False, device=device)[0].to(device).reshape(1, H, W, 2).half()
    ft_per_vertex = torch.zeros((len(mesh_vertices), FEATURE_DIMS), device=device).half()
    ft_per_vertex_count = torch.zeros((len(mesh_vertices), 1), device=device).half()
    # iterate views
    for idx in range(len(batched_renderings)):
        img = (batched_renderings[idx, :, :, :3].cpu().numpy() * 255).astype(np.uint8).squeeze()
        output_image = Image.fromarray(img)
        aligned_dino_features = get_dino_features(device, dino_model, output_image, grid, idx)
        aligned_features = aligned_dino_features
        # simplified: assign the same features to all vertices visible in this view
        # here we approximate by adding features to all vertices
        ft_per_vertex += aligned_features[0].T[: len(mesh_vertices), :].half()
        ft_per_vertex_count += 1
    ft_per_vertex = ft_per_vertex / ft_per_vertex_count.clamp(min=1)
    return ft_per_vertex


def compute_shape_dino_features(verts, faces):
    if not PYTORCH3D_AVAILABLE:
        raise RuntimeError('pytorch3d not available in environment')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = get_dino_model(device)
    verts = verts.clone().detach().to(dtype=torch.float32)
    faces = faces.clone().detach().to(dtype=torch.int64)
    verts_rgb = torch.ones_like(verts)[None] * 0.8
    from pytorch3d.renderer import Textures
    textures = Textures(verts_rgb=verts_rgb)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    mesh = mesh.to(device)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex(device=device, dino_model=dino_model, mesh=mesh, mesh_vertices=mesh_vertices, num_views=32, H=256, W=256)
    return features.cpu()


def get_shape_dino_features(verts, faces, cache_dir=None):
    # simple wrapper (no caching implemented here)
    return compute_shape_dino_features(verts, faces)
