import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import binary_dilation, convolve, distance_transform_edt
from utils.clean_textures import clean_textures


def load_obj_uvs(filepath):
    vertices, uvs, faces, face_uvs = [], [], [], []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:4])))
            elif line.startswith('vt '):
                u, v = map(float, line.strip().split()[1:3])
                uvs.append([u, 1.0 - v])  # Flip V
            elif line.startswith('f '):
                verts = line.strip().split()[1:]
                face, uv_face = [], []
                for v in verts:
                    v_parts = v.split('/')
                    face.append(int(v_parts[0]) - 1)
                    uv_face.append(int(v_parts[1]) - 1)
                faces.append(face)
                face_uvs.append(uv_face)
    return np.array(vertices), np.array(uvs), np.array(faces), np.array(face_uvs)


def barycentric_coords(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    if denom == 0:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def is_inside_triangle(u, v, w, margin=0.01):
    return all(-margin <= val <= 1 + margin for val in (u, v, w))


def bilinear_sample(image, uv):
    h, w = image.shape[:2]
    u, v = uv[0] * (w - 1), uv[1] * (h - 1)
    x0, y0 = int(np.floor(u)), int(np.floor(v))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    dx, dy = u - x0, v - y0
    top = (1 - dx) * image[y0, x0] + dx * image[y0, x1]
    bot = (1 - dx) * image[y1, x0] + dx * image[y1, x1]
    return ((1 - dy) * top + dy * bot).astype(np.uint8)


def dilate_texture_with_blending(texture, mask, iterations=2):
    h, w, c = texture.shape
    tex = texture.copy()
    for _ in range(iterations):
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        dilate_candidates = convolve(mask.astype(np.uint8), kernel, mode='constant') > 0
        dilate_candidates &= ~mask
        for y in range(h):
            for x in range(w):
                if dilate_candidates[y, x]:
                    neighbors = [tex[ny, nx] for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]
                                 if 0 <= (ny := y + dy) < h and 0 <= (nx := x + dx) < w and mask[ny, nx]]
                    if neighbors:
                        tex[y, x] = np.mean(neighbors, axis=0).astype(np.uint8)
                        mask[y, x] = True
    return tex


def reproject_texture(old_uvs, new_uvs, faces, old_face_uvs, new_face_uvs, old_tex, res=1024):
    h, w = res, res
    new_tex = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)

    for i in tqdm(range(len(faces))):
        old_uv = old_uvs[old_face_uvs[i]]
        new_uv = new_uvs[new_face_uvs[i]]
        new_pix = new_uv * [w - 1, h - 1]
        min_x, max_x = int(np.floor(np.min(new_pix[:, 0]))), int(np.ceil(np.max(new_pix[:, 0])))
        min_y, max_y = int(np.floor(np.min(new_pix[:, 1]))), int(np.ceil(np.max(new_pix[:, 1])))
        min_x, max_x = max(min_x, 0), min(max_x, w - 1)
        min_y, max_y = max(min_y, 0), min(max_y, h - 1)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                bc = barycentric_coords(np.array([x, y]), *new_pix)
                if bc is None or not is_inside_triangle(*bc):
                    continue
                uv_old = bc[0] * old_uv[0] + bc[1] * old_uv[1] + bc[2] * old_uv[2]
                new_tex[y, x] = bilinear_sample(old_tex, uv_old)
                mask[y, x] = True

    return dilate_texture_with_blending(new_tex, mask, iterations=2)


if __name__ == "__main__":
    SOURCE_OBJ = "mesh/smpl_deformed/smpl_deformed2.obj"
    TARGET_OBJ = "mesh/source8/source8.obj"
    INPUT_TEXTURE_DIR = "./smplitex"
    OUTPUT_TEXTURE_DIR = "./source8_textures"

    _, old_uvs, faces, old_face_uvs = load_obj_uvs(SOURCE_OBJ)
    _, new_uvs, _, new_face_uvs = load_obj_uvs(TARGET_OBJ)

    texture_files = [f for f in sorted(os.listdir(INPUT_TEXTURE_DIR)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    os.makedirs(OUTPUT_TEXTURE_DIR, exist_ok=True)

    for tex_name in tqdm(texture_files, desc="Reprojecting Textures"):
        tex_path = os.path.join(INPUT_TEXTURE_DIR, tex_name)    
        out_path = os.path.join(OUTPUT_TEXTURE_DIR, tex_name)
        old_texture = np.array(Image.open(tex_path).convert("RGB"))

        new_texture = reproject_texture(
            old_uvs, new_uvs, faces,
            old_face_uvs, new_face_uvs,
            old_texture, res=512
        )

        Image.fromarray(new_texture).save(out_path)

    print(f"\n✅ Finished reprojecting {len(texture_files)} textures to {OUTPUT_TEXTURE_DIR}")
