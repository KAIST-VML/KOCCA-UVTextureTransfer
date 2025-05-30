import numpy as np
import cv2
from scipy.ndimage import map_coordinates


def parse_obj_with_uv_per_corner(obj_path):
    faces = []
    uv_coords = []
    with open(obj_path, 'r') as f:
        verts = []
        uvs = []
        for line in f:
            if line.startswith("v "):
                verts.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith("vt "):
                uvs.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                face = []
                uv_face = []
                for part in parts:
                    v_idx, vt_idx = map(int, part.split('/')[:2])
                    face.append(v_idx - 1)
                    uv_face.append(uvs[vt_idx - 1])
                faces.append(uv_face)  # Store UVs directly
    return np.array(faces)  # shape (F, 3, 2)


def render_uv_mask_from_white_uvs(obj_path, mask_path, output_path, texture_res=1024, threshold=200):
    uv_faces = parse_obj_with_uv_per_corner(obj_path)  # shape (F, 3, 2)
    rough_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    assert rough_mask.shape == (texture_res, texture_res)

    # Prepare coordinates for sampling
    uv_flat = uv_faces.reshape(-1, 2)
    uv_sample_coords = uv_flat.copy()
    uv_sample_coords[:, 0] *= (texture_res - 1)
    uv_sample_coords[:, 1] = (1 - uv_sample_coords[:, 1]) * (texture_res - 1)

    sampled_values = map_coordinates(
        rough_mask, [uv_sample_coords[:, 1], uv_sample_coords[:, 0]],
        order=1, mode='nearest'
    )

    white_mask = sampled_values >= threshold
    white_mask = white_mask.reshape(-1, 3)

    # Keep only triangles where all three UVs landed on white
    keep_face = np.all(white_mask, axis=1)
    white_uv_faces = uv_faces[keep_face]

    # Rasterize
    uv_mask = np.zeros((texture_res, texture_res), dtype=np.uint8)
    for tri_uv in white_uv_faces:
        pts = (tri_uv * (texture_res - 1)).astype(np.int32)
        pts[:, 1] = texture_res - 1 - pts[:, 1]  # Flip Y
        cv2.fillConvexPoly(uv_mask, pts.reshape((-1, 1, 2)), 255)

    cv2.imwrite(output_path, uv_mask)
    print(f"[DONE] Mask rendered from UVs to {output_path}")

render_uv_mask_from_white_uvs(
    obj_path="aaron_005.obj",
    mask_path="mask3.png",
    output_path="clean_face_mask.png",
    texture_res=1024,
    threshold=200
)
