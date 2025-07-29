# UV 기반 텍스처 전이를 통한 모델 전환
Mesh-agnostic UV texture transfer.

## Setup
### Create conda environment
```
conda env create -f environment.yml
conda activate uvtt
```

### Install Pytorch3D
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

## Usage
### Compute diffusion 3D features
```
python get_features.py 'data/source/mesh'
```
This will compute the vertex features for each mesh within 'data/source/mesh' and save them in 'data/source/features'. The number of meshes within the folder is arbitrary. You can put the whole body mesh or separate them into head and body like I did. The prompt will be based on the name of the obj file (human_head.obj = 'human head').

The seed for the diffusion model is currently fixed (check diff3f/diffusion.py). 

```
python get_features.py 'data/target/mesh'
```
Do the same for the target mesh.

The names 'source' and 'target' are arbitrary. Just match the following directory structure:

```
.
├── data/
│   ├── source/
│   │   ├── mesh/
│   │   └── features/
│   └── target/
│       ├── mesh/
│       └── features/
```

### Transfer textures
```
python transfer_textures.py data/source data/target
```
Provided that you computed the diffusion 3D features, this code will transfer the source textures from the source UV to the target UV.

## Evaluation
### Creating ground truth texture output
While this method is mesh agnostic, creating ground truth target texture from the source only works when the mesh connectivity is identical (check the code for required inputs)
```
python utils/reproject_textures.py
```

### Calculate Fréchet inception distance (FID)
```
python evaluate_textures.py data/target/textures output/source2target
```
