# Imports
import numpy as np
import os
import sys
import argparse
import torch

# From diff3f
sys.path.append('diff3f')
from diff3f import get_features_per_vertex
from utils import convert_mesh_container_to_torch_mesh, cosine_similarity, double_plot, get_colors, generate_colors
from dataloaders.mesh_container import MeshContainer
from diffusion import init_pipe
from dino import init_dino
from functional_map import compute_surface_map


def compute_features(device, pipe, dino_model, m, prompt):
    mesh = convert_mesh_container_to_torch_mesh(m, device=device, is_tosca=False)
    mesh_vertices = mesh.verts_list()[0]
    features = get_features_per_vertex(
        device=device,
        pipe=pipe, 
        dino_model=dino_model,
        mesh=mesh,
        prompt=prompt,
        mesh_vertices=mesh_vertices,
        num_views=100,
        H=512,
        W=512,
        tolerance=0.004,
        num_images_per_prompt=1,
        use_normal_map=True,
    )
    return features.cpu()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_dir", type=str, help="Path to mesh directory (e.g., data/source/mesh)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_dir = args.mesh_dir
    data_dir = os.path.dirname(mesh_dir)  # e.g., from data/target/mesh → data/target

    # Load models
    pipe = init_pipe(device)
    dino_model = init_dino(device)

    os.makedirs(os.path.join(data_dir, 'features'), exist_ok=True)
    for mesh_name in os.listdir(mesh_dir):
        mesh_path = os.path.join(mesh_dir, mesh_name)
        mesh = MeshContainer().load_from_file(mesh_path)
        prompt = mesh_name.replace('.obj', '').replace('_', ' ')
        features = compute_features(device, pipe, dino_model, mesh, prompt)
        out_path = os.path.join(data_dir, "features", mesh_name.replace('.obj', '.npy'))
        np.save(out_path, features)