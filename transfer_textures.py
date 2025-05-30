import argparse
import numpy as np
import os
import shutil
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch3d.io import load_obj
from tqdm import tqdm
from torchvision import transforms
from utils.clean_textures import clean_textures
from utils.texture_transfer import (
    compute_uniform_laplacian,
    smooth_features,
    rasterize_features,
    match_target_to_source
)


def transfer_textures(source_dir, output_dir, best_match_flat_indices, device='cuda'):
    to_tensor = transforms.ToTensor()  # (C, H, W) in [0, 1]
    to_pil = transforms.ToPILImage()

    filenames = sorted(f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    for fname in tqdm(filenames, desc="Transferring textures"):
        # Load source image
        img_path = os.path.join(source_dir, fname)
        img = Image.open(img_path).convert('RGB')
        img_tensor = to_tensor(img).permute(1, 2, 0).to(device)  # (H, W, 3)

        H, W, _ = img_tensor.shape
        assert H * W == best_match_flat_indices.shape[0], f"Mismatch in dimensions: {fname}"

        # Transfer
        source_texture_flat = img_tensor.view(-1, 3)
        target_texture_flat = source_texture_flat[best_match_flat_indices]
        target_texture = target_texture_flat.view(H, W, 3).clamp(0, 1)

        # Save
        output_path = os.path.join(output_dir, fname)

        if os.path.exists(output_path):
            existing_img = Image.open(output_path).convert('RGB')
            existing_tensor = to_tensor(existing_img).permute(1, 2, 0).to(device)
            mask = (target_texture > 1e-6).any(dim=-1)

            existing_tensor[mask] = target_texture[mask]

            target_texture = existing_tensor

        to_pil(target_texture.permute(2, 0, 1).cpu()).save(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str, help='Source directory')
    parser.add_argument('target_dir', type=str, help='Target directory')
    args = parser.parse_args()

    source_name = os.path.basename(args.source_dir)
    target_name = os.path.basename(args.target_dir)
    output_dir = f'output/{source_name}2{target_name}'

    # Make sure output folder doesn't exist at first
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_features_dir = f"data/{source_name}/features"
    target_features_dir = f"data/{target_name}/features"
    
    for features_name in os.listdir(source_features_dir):
        # Source
        source_features_path = os.path.join(source_features_dir, features_name)
        source_features = torch.from_numpy(np.load(source_features_path)).to(device)
        source_mesh_path = f"data/{source_name}/mesh/{features_name.replace('.npy', '.obj')}"
        source_verts, source_faces, source_aux = load_obj(source_mesh_path, load_textures=True)
        source_vert_uvs = source_aux.verts_uvs
        source_face_uvs = source_faces.textures_idx
        source_face_verts = source_faces.verts_idx.to(device)
        source_L = compute_uniform_laplacian(source_verts.shape[0], source_face_verts)
        source_features = smooth_features(source_features.float(), source_L, lambd=0.01, iterations=200).half()
        source_map = rasterize_features(
            verts_uv=source_vert_uvs,
            faces_uv_idx=source_face_uvs,
            faces_vert_idx=source_face_verts,
            vert_features=source_features,
            resolution=512,
            faces_per_pixel=10,
            chunk_size=128,
            device=device
        )

        # Target
        target_features_path = os.path.join(target_features_dir, features_name)
        target_features = torch.from_numpy(np.load(target_features_path)).to(device)
        target_mesh_path = f"data/{target_name}/mesh/{features_name.replace('.npy', '.obj')}"
        target_verts, target_faces, target_aux = load_obj(target_mesh_path, load_textures=True)
        target_vert_uvs = target_aux.verts_uvs
        target_face_uvs = target_faces.textures_idx
        target_face_verts = target_faces.verts_idx.to(device)
        target_L = compute_uniform_laplacian(target_verts.shape[0], target_face_verts)
        target_features = smooth_features(target_features.float(), target_L, lambd=0.01, iterations=200).half()
        target_map = rasterize_features(
            verts_uv=target_vert_uvs,
            faces_uv_idx=target_face_uvs,
            faces_vert_idx=target_face_verts,
            vert_features=target_features,
            resolution=512,
            faces_per_pixel=10,
            chunk_size=128,
            device=device
        )

        # Match
        source_n = F.normalize(source_map, dim=1)
        target_n = F.normalize(target_map, dim=1)
        best_match_flat_indices = match_target_to_source(target_n, source_n, chunk_size=2048)

        # Transfer textures
        transfer_textures(
            source_dir = f'data/{source_name}/textures',
            output_dir = output_dir,
            best_match_flat_indices = best_match_flat_indices,
            device = device
        )

    clean_textures(
        input_dir=output_dir,
        output_dir=output_dir,
        uv_mask_path=f'data/{target_name}/mask/mask.png',
        dilate_iters=1,
        resize_textures=True
    )
