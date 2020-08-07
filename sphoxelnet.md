# Sphoxelnet

A one-stage 3d object detection network with spherical voxelization.

## Usage

work directory
```
cd tools
```

training

```
python train.py --cfg_file cfgs/kitti_models/<model.yaml> --batch_size 8 --epoch 50 --extra_tag <tag>
```

## Models

- `sphoxelnet_simple.yaml`

    sphoxelnet with spconv, for testing
- `sphoxelnet.yaml`

    sphoxelnet with sphconv

## dependency

- `sphconv`

## visualization

go to work directory

```
cd tools
```

start backend

```
python ../kittiviewer/backend/main.py main --port=16666
```

start frontend

```
cd ../kittiviewer/frontend/
python -m http.server
```

