# PCL compression
https://pcl.readthedocs.io/projects/tutorials/en/latest/compression.html#octree-compression
## Compression Profiles:
```
LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR 1 cubic centimeter resolution, no color, fast online encoding
LOW_RES_ONLINE_COMPRESSION_WITH_COLOR 1 cubic centimeter resolution, color, fast online encoding
MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR 5 cubic millimeter resolution, no color, fast online encoding
MED_RES_ONLINE_COMPRESSION_WITH_COLOR 5 cubic millimeter resolution, color, fast online encoding
HIGH_RES_ONLINE_COMPRESSION_WITHOUT_COLOR 1 cubic millimeter resolution, no color, fast online encoding
HIGH_RES_ONLINE_COMPRESSION_WITH_COLOR 1 cubic millimeter resolution, color, fast online encoding
LOW_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR 1 cubic centimeter resolution, no color, efficient offline encoding
LOW_RES_OFFLINE_COMPRESSION_WITH_COLOR 1 cubic centimeter resolution, color, efficient offline encoding
MED_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR 5 cubic millimeter resolution, no color, efficient offline encoding
MED_RES_OFFLINE_COMPRESSION_WITH_COLOR 5 cubic millimeter resolution, color, efficient offline encoding
HIGH_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR 1 cubic millimeter resolution, no color, efficient offline encoding
HIGH_RES_OFFLINE_COMPRESSION_WITH_COLOR 1 cubic millimeter resolution, color, efficient offline encoding
MANUAL_CONFIGURATION enables manual configuration for advanced parametrization
```
## Advanced parametrization:
```
OctreePointCloudCompression (compression_Profiles_e compressionProfile_arg,
                             bool showStatistics_arg,
                             const double pointResolution_arg,
                             const double octreeResolution_arg,
                             bool doVoxelGridDownDownSampling_arg,
                             const unsigned int iFrameRate_arg,
                             bool doColorEncoding_arg,
                             const unsigned char colorBitResolution_arg
                            )
```
compressionProfile_arg: This parameter should be set to MANUAL_CONFIGURATION for enabling advanced parametrization.
showStatistics_arg: Print compression related statistics to stdout.
pointResolution_arg: Define coding precision for point coordinates. This parameter should be set to a value below the sensor noise.
octreeResolution_arg: This parameter defines the voxel size of the deployed octree. A lower voxel resolution enables faster compression at, however, decreased compression performance. This enables a trade-off between high frame/update rates and compression efficiency.
doVoxelGridDownDownSampling_arg: If activated, only the hierarchical octree data structure is encoded. The decoder generated points at the voxel centers. In this way, the point cloud becomes downsampled during compression while achieving high compression performance.
iFrameRate_arg: The point cloud compression scheme differentially encodes point clouds. In this way, differences between the incoming point cloud and the previously encoded pointcloud is encoded in order to achieve maximum compression performance. The iFrameRate_arg allows to specify the rate of frames in the stream at which incoming point clouds are not differentially encoded (similar to I/P-frames in video coding).
doColorEncoding_arg: This option enables color component encoding.
colorBitResolution_arg: This parameter defines the amount of bits per color component to be encoded.

