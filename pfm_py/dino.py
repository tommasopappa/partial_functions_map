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

FEATURE_DIMS = 768
VERTEX_GPU_LIMIT = 35000
_DINO_MODEL = None


def get_dino_model(device):
    global _DINO_MODEL
    if _DINO_MODEL is None:
        _DINO_MODEL = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        _DINO_MODEL = _DINO_MODEL.to(device).eval()
    return _DINO_MODEL


@torch.no_grad()
def get_dino_features(device, dino_model, img, grid):
    patch_size = 14
    transform = tfs.Compose([
        tfs.Resize((518, 518)),
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)[:3].unsqueeze(0).to(device)
    features = dino_model.get_intermediate_layers(img, n=1)[0].half()
    h, w = img.shape[2] // patch_size, img.shape[3] // patch_size
    features = features.reshape(-1, h, w, FEATURE_DIMS).permute(0, 3, 1, 2)
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


def batch_render(device, mesh, num_views, H, W):
    bbox = mesh.get_bounding_boxes()
    bbox_min, bbox_max = bbox.min(dim=-1).values[0], bbox.max(dim=-1).values[0]
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt(((bbox_max - bbox_min) ** 2).sum()) * 0.65

    steps = int(math.ceil(math.sqrt(num_views)))
    end = 360 - 360 / steps
    elevation = torch.linspace(0, end, steps).repeat(steps)[:num_views]
    azimuth = torch.linspace(0, end, steps).repeat_interleave(steps)[:num_views]

    R, T = look_at_view_transform(dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center.unsqueeze(0))
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


def compute_dino_features(verts, faces, num_views=100, H=512, W=512, tolerance=0.004):
    if not PYTORCH3D_AVAILABLE:
        raise RuntimeError('pytorch3d not available')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino_model = get_dino_model(device)

    verts = verts.clone().detach().float()
    faces = faces.clone().detach().long()
    textures = Textures(verts_rgb=torch.ones_like(verts)[None] * 0.8)
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures).to(device)
    mesh_vertices = mesh.verts_list()[0]

    # compute ball radius
    if len(mesh_vertices) > VERTEX_GPU_LIMIT:
        samples = torch.randperm(len(mesh_vertices))[:10000]
        max_dist = torch.cdist(mesh_vertices[samples], mesh_vertices[samples]).max()
    else:
        max_dist = torch.cdist(mesh_vertices, mesh_vertices).max()
    ball_radius = max_dist * tolerance

    images, camera, depth = batch_render(device, mesh, num_views, H, W)
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

    # average where we have features
    has_feats = ft_count[:, 0] != 0
    ft_per_vertex[has_feats] /= ft_count[has_feats]

    # fill missing with nearest neighbor
    missing = ~has_feats
    n_missing = missing.sum().item()
    if n_missing > 0:
        print(f"Warning: {n_missing} vertices missing features, using nearest neighbor")
        dists = torch.cdist(mesh_vertices[missing], mesh_vertices[has_feats])
        nearest = dists.argmin(dim=1)
        ft_per_vertex[missing] = ft_per_vertex[has_feats][nearest]

    return ft_per_vertex.detach(), n_missing


# convenience aliases for backward compatibility
def get_shape_dino_features(verts, faces, num_views=100, H=512, W=512, tolerance=0.004):
    feats, _ = compute_dino_features(verts, faces, num_views, H, W, tolerance)
    return feats

compute_shape_dino_features = get_shape_dino_features
