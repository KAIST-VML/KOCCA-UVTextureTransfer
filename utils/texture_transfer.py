import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OrthographicCameras,
    MeshRasterizer,
    RasterizationSettings,
)


def compute_uniform_laplacian(n_verts, faces):
    I = torch.cat([faces[:, 0], faces[:, 1], faces[:, 2]])
    J = torch.cat([faces[:, 1], faces[:, 2], faces[:, 0]])
    edges = torch.cat([torch.stack([I, J], dim=0), torch.stack([J, I], dim=0)], dim=1)
    values = torch.ones(edges.shape[1], device=faces.device)
    W = torch.sparse_coo_tensor(edges, values, (n_verts, n_verts)).coalesce()
    deg = torch.sparse.sum(W, dim=1).to_dense()
    diag_idx = torch.arange(n_verts, device=faces.device)
    D = torch.sparse_coo_tensor(torch.stack([diag_idx, diag_idx]), deg, size=(n_verts, n_verts))
    L = D - W
    return L.coalesce()


def smooth_features(features, L, lambd=0.1, iterations=1):
    for _ in range(iterations):
        features = features - lambd * torch.sparse.mm(L, features)
    return features


def rasterize_features(
    verts_uv, faces_uv_idx, faces_vert_idx, vert_features,
    resolution=512, faces_per_pixel=3, chunk_size=128,
    flip_u=True, flip_v=False, device='cuda'
):
    uv = verts_uv.clone()
    if flip_u:
        uv[:, 0] = 1.0 - uv[:, 0]
    if flip_v:
        uv[:, 1] = 1.0 - uv[:, 1]

    verts_uv_3d = torch.cat([uv, torch.zeros_like(uv[:, :1])], dim=-1).to(device)
    mesh_uv = Meshes(verts=[verts_uv_3d], faces=[faces_uv_idx.to(device)])
    face2verts = faces_vert_idx.to(device)
    C = vert_features.shape[-1]
    output = torch.zeros((resolution, resolution, C), device=device)

    cameras = OrthographicCameras(
        device=device,
        focal_length=torch.tensor([[2.0, 2.0]], device=device),
        T=torch.tensor([[-0.5, -0.5, 1.0]], device=device)
    )

    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=faces_per_pixel,
            bin_size=None
        )
    )

    fragments = rasterizer(mesh_uv)
    pix_to_face_full = fragments.pix_to_face[0]
    bary_coords_full = fragments.bary_coords[0]
    zbuf_full = fragments.zbuf[0]

    for y0 in range(0, resolution, chunk_size):
        for x0 in range(0, resolution, chunk_size):
            h = min(chunk_size, resolution - y0)
            w = min(chunk_size, resolution - x0)
            pix_to_face = pix_to_face_full[y0:y0+h, x0:x0+w]
            bary_coords = bary_coords_full[y0:y0+h, x0:x0+w]
            face_idx_clamped = pix_to_face.clone()
            face_idx_clamped[face_idx_clamped < 0] = 0
            vert_indices = face2verts[face_idx_clamped]
            f0 = vert_features[vert_indices[..., 0]]
            f1 = vert_features[vert_indices[..., 1]]
            f2 = vert_features[vert_indices[..., 2]]
            interp = f0 * bary_coords[..., 0:1] + f1 * bary_coords[..., 1:2] + f2 * bary_coords[..., 2:3]
            valid = pix_to_face >= 0
            interp[~valid] = 0
            valid_counts = valid.sum(dim=-1, keepdim=True).clamp(min=1)
            blended = interp.sum(dim=2) / valid_counts
            output[y0:y0+h, x0:x0+w] = blended

    return output.view(-1, C)


def match_target_to_source(tgt_feats, src_feats, chunk_size=1024):
    best_match_idx = []
    for i in range(0, tgt_feats.size(0), chunk_size):
        tgt_chunk = tgt_feats[i:i+chunk_size]
        sim = torch.matmul(tgt_chunk, src_feats.T)
        idx = sim.argmax(dim=1)
        best_match_idx.append(idx)
    return torch.cat(best_match_idx, dim=0)