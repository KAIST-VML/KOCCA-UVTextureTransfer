import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F


def load_uv_mask(uv_mask_path):
    uv_mask_img = Image.open(uv_mask_path).convert('L')  # grayscale
    uv_mask = np.array(uv_mask_img).astype(np.float32) / 255.0
    uv_mask = (uv_mask > 0.5).astype(np.float32)
    return torch.from_numpy(uv_mask)  # (H, W) tensor


def load_texture(filepath):
    img = Image.open(filepath).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0  # (H, W, 3)
    return img_tensor


def save_texture(tensor, filepath):
    tensor = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()
    img = Image.fromarray(tensor)
    img.save(filepath)


def dilate_texture_pytorch(texture, mask, iterations=2):
    texture = texture.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    mask = mask.unsqueeze(0).unsqueeze(0).float()    # (1, 1, H, W)

    for _ in range(iterations):
        dilated_mask = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        new_valid = (dilated_mask > 0.5) & (mask == 0)

        for c in range(texture.shape[1]):
            channel = texture[:, c:c+1, :, :]
            dilated_color = F.max_pool2d(channel * mask, kernel_size=3, stride=1, padding=1)
            channel[new_valid] = dilated_color[new_valid]
            texture[:, c:c+1, :, :] = channel

        mask = dilated_mask

    return texture.squeeze(0).permute(1, 2, 0)  # (H, W, 3)


def clean_texture(texture, uv_mask, dilate_iters=2):
    masked_texture = texture * uv_mask.unsqueeze(-1)
    if dilate_iters > 0:
        masked_texture = dilate_texture_pytorch(masked_texture, uv_mask, iterations=dilate_iters)
    return masked_texture


def clean_textures(
    input_dir,
    output_dir,
    uv_mask_path,
    dilate_iters=1,
    resize_textures=True,
):
    os.makedirs(output_dir, exist_ok=True)
    uv_mask = load_uv_mask(uv_mask_path)  # (H, W)
    uv_h, uv_w = uv_mask.shape

    texture_files = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for fname in tqdm(texture_files, desc="Cleaning Textures"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        tex = load_texture(input_path)

        if resize_textures and (tex.shape[0] != uv_h or tex.shape[1] != uv_w):
            tex = torch.nn.functional.interpolate(
                tex.permute(2, 0, 1).unsqueeze(0),
                size=(uv_h, uv_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0)

        cleaned = clean_texture(tex, uv_mask, dilate_iters=dilate_iters)
        save_texture(cleaned, output_path)


if __name__ == "__main__":

    input_dir = 'data/target/textures'
    output_dir = 'data/target/textures'
    uv_mask_path = 'data/target/mask/mask.png'
    dilate_iters = 1
    resize_textures = True

    clean_textures(
        input_dir=input_dir,
        output_dir=output_dir,
        uv_mask_path=uv_mask_path,
        dilate_iters=dilate_iters,
        resize_textures=resize_textures,
    )