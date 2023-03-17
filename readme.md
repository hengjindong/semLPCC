# Semantic-based LiDAR Point Cloud Compression (S-LPCC)

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

* JPEG "ffmpeg -i input_path -vcodec libopenjpeg output_path"
* PNG numpngw
* Jon2016 FLIF
* PNGOut pngout
* WebP
* BPG bpgenc
* APNG

### Draco 

list_draco_qp: Compression quality grade default [16, 15, 14, 13, 12, 11, 10, 9]

```python
python test_comparison_draco.py
```

### MPGE G-PCC

list_mpeg_qp: Compression quality grade default [1000, 500, 250, 100, 50, 20, 15, 10, 5]
The G-PCC compressed point cloud is an integer, so the choice of the original point cloud will be low when it comes to quality assessment.

```python
python test_comparison_draco.py
```
### PCL

list_draco_qp: Compression quality grade default [0.7, 0.5, 0.2]

```python
python test_comparison_draco.py
```

## 帧内消融实验

* Depth R
* Depth+Insatance-semantic R+I
* Depth+Ground Simulation R+GS
* Depth+Insatance-semantic+Ground Simulation R+I+GS

dataset_path 数据集位置
seq_name 测试序列 00-11
seq_len 测试帧数量 100
depth_width depth_hight 深度图宽度、高度 2048 64
loss_bit_range 深度图有损比特数量 0-8

### 深度图 R

```python
python Intra_R.py
```

### 深度图+语义区分 R+I

```python
python Intra_R_I.py
```

### 深度图+地面模拟 R+GS

```python
python Intra_R_GS.py
```

### 深度图+语义区分+地面模拟 R+I+GS

```python
python Intra_R_I_GS.py
```

## 帧间消融实验

* 运动估计 ME
* 运动估计+语义区分 ME+I
* 运动估计+边缘残差 ME+B
* 运动估计+语义区分+边缘残差 ME+I+B

dataset_path 数据集位置
seq_name 测试序列 00-11
seq_len 测试帧数量 100
depth_width depth_hight 深度图宽度、高度 2048 64
loss_bit_range 深度图有损比特数量 0-8
scene_interval 帧间间隔 1
threshold 运动估计中icp参数 1
loss_bit_range 深度图有损比特数量 0-8

### 运动估计 ME

border_num 边缘残差计算高度 64
```python
python Inter_ME_B.py
```

### 运动估计+语义区分 ME+I

border_num 边缘残差计算高度 64
```python
python Inter_ME_I_B.py
```

### 运动估计+边缘残差 ME+B

border_num 边缘残差计算高度 30
```python
python Inter_ME_B.py
```

### 运动估计+语义区分+边缘残差 ME+I+B

border_num 边缘残差计算高度 30
```python
python Inter_ME_I_B.py
```

## 分割精度实验

使用Cylinder3D进行语义分割
对压缩后的数据进行格式处理

1. 保存深度图对应的label
```python
python Perc_label_save.py
```

2. 保存不同精度深度图对应的点云
```python
python Perception.py
```

3. 保存不同精度的语义区分深度图对应的点云
```python
Perception_I.py
```

## 定位时间实验

直接进行icp点云配准，统计时间
```python
Location.py
```

对进行语义区分后的点云进行icp点云配准，统计时间
```python
python Location_I.py
```
