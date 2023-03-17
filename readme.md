# Semantic-based LiDAR Point Cloud Compression (semLPCC)

## env list

```
python 3.8.12
h5py 3.6.0
numpy 1.21.2
open3d 0.13.0
plyfile 0.7.4
torch 1.7.1
pytorch3d 0.6.0
spconv 1.2.1
numpngw 0.1.2
```

## intra compression

dataset_path: your semanticKITTI dataset path
seq_name: test seq default 00-11
seq_len: test frame number default 100
depth_width&depth_hight: depth width&depth default 2048 64
loss_bit_range: depth lossly bit 0 - 12

```python
python Intra_R_I_GS.py
```

## inter compression

dataset_path: your semanticKITTI dataset path
seq_name: test seq default 00-11
seq_len: test frame number default 100
depth_width&depth_hight: depth width&depth default 2048 64
scene_interval: farme interval default 1
threshold: move estimation (ICP) parameters default 1
border_num: border residual calculated height default 30
loss_bit_range: depth lossly bit 0 - 12

```python
python Inter_ME_I_B.py
```

## Comparison method

### Image compression comparison

Adding an image save command to **Intra_R** to save depth image. (default numpngw uint16)
Compression of the saved depth image using image compression software.

```python
python Intra_R.py
```

* JPEG ffmpeg
* PNG numpngw
* Jon2016 [FLIF](https://github.com/FLIF-hub/FLIF)
* [PNGOut](http://www.ardfry.com/pngoutwin)
* [WebP](http://developers.google.com/speed/webp)
* [BPG](http://bellard.org/bpg)
* [APNG](https://sourceforge.net/projects/apngasm)

### [Draco](htttps://github.com/google/draco) 

list_draco_qp: Compression quality grade default [16, 15, 14, 13, 12, 11, 10, 9]

```python
python test_comparison_draco.py
```

### MPGE [G-PCC](https://github.com/MPEGGroup/mpeg-pcc-tmc13)

list_mpeg_qp: Compression quality grade default [1000, 500, 250, 100, 50, 20, 15, 10, 5]
The G-PCC compressed point cloud is an integer, so the choice of the original point cloud will be low when it comes to quality assessment.

```python
python test_comparison_draco.py
```
### [PCL](https://github.com/PointCloudLibrary/pcl)

list_draco_qp: Compression quality grade default [0.7, 0.5, 0.2]

```python
python test_comparison_draco.py
```

## Intra ablation experiments

* Depth R
* Depth+Semantic R+I
* Depth+Ground Simulation R+GS
* Depth+Semantic+Ground Simulation R+I+GS

dataset_path: your semanticKITTI dataset path
seq_name: test seq default 00-11
seq_len: test frame number default 100
depth_width&depth_hight: depth width&depth default 2048 64
loss_bit_range: depth lossly bit 0 - 12

### Depth R

```python
python Intra_R.py
```

### Depth+Semantic R+I

```python
python Intra_R_I.py
```

### Depth+Ground Simulation R+GS

```python
python Intra_R_GS.py
```

### Depth+Semantic+Ground Simulation R+I+GS

```python
python Intra_R_I_GS.py
```

## Inter ablation experiments

* Move Estimation ME
* Move Estimation+Semantic ME+I
* Move Estimation+Border Residuals ME+B
* Move Estimation+Semantic+Border Residuals ME+I+B

dataset_path: your semanticKITTI dataset path
seq_name: test seq default 00-11
seq_len: test frame number default 100
depth_width&depth_hight: depth width&depth default 2048 64
scene_interval: farme interval default 1
threshold: move estimation (ICP) parameters default 1
border_num: border residual calculated height default 30
loss_bit_range: depth lossly bit 0 - 12

### Move Estimation ME

border_num: 64
```python
python Inter_ME_B.py
```

### Move Estimation+Semantic ME+I

border_num: 64
```python
python Inter_ME_I_B.py
```

### Move Estimation+Border Residuals ME+B

border_num: 30
```python
python Inter_ME_B.py
```

### Move Estimation+Semantic+Border Residuals ME+I+B

border_num: 30
```python
python Inter_ME_I_B.py
```

## Segmentation accuracy experiments

Semantic segmentation using **Cylinder3D**.
Processing of compressed depth data.

1. save depth corresponding label
```python
python Perc_label_save.py
```

2. save of depth point clouds with different lossy bit.
```python
python Perception.py
```

3. save of after semantic point clouds with different lossy bit.
```python
Perception_I.py
```

## Loaction cost time experiments

Direct icp point cloud alignment (based on the icp method in open3d) statistical time spent.
```python
Location.py
```

after semantic point cloud icp statistical cost time.
```python
python Location_I.py
```
