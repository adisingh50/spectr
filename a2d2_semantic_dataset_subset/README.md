## Semantic Segmentation

The semantic segmentation subset contains 41,277 frames. Each frame
contains the following items:

- RGB image
- 3D point cloud
- annotated semantic segmentation label

Frames are grouped in 23 different scenes, with each scene in its own
directory. Scene directory names denote the time and date of the
recording in 'YYYYMMDD_hhmmss' format. Each scene has the following
three subdirectories:

- 'camera': images and json info files
- 'lidar': 3D point clouds
- 'label': pixelwise semantic segmentation labels

Each of these directories contains further subdirectories for each
camera. There are six cameras available in the vehicle, corresponding
to the following subdirectories:

- 'cam_front_center'
- 'cam_front_left'
- 'cam_front_right'
- 'cam_side_left'
- 'cam_side_right'
- 'cam_rear_center'

These are the filename formats for the items of a single frame:

RGB image           : YYMMDDDDhhmmss_camera_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].png
info                : YYMMDDDDhhmmss_camera_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].json
LiDAR point cloud   : YYMMDDDDhhmmss_lidar_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].npz
semantic label image: YYMMDDDDhhmmss_label_[frontcenter|frontleft|frontright|sideleft|sideright|rearcenter]_[ID].png

For example, a frame with ID 1617 from a scene recorded on 2018-08-07
14:50:28 from the front center camera would consist of the following
items:

RGB image           : 20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000001617.png
info                : 20180807_145028/camera/cam_front_center/20180807145028_camera_frontcenter_000001617.json
LiDAR point cloud   : 20180807_145028/lidar/cam_front_center/20180807145028_lidar_frontcenter_000001617.npz
semantic label image: 20180807_145028/label/cam_front_center/20180807145028_label_frontcenter_000001617.png

For further explanations regarding the format of each of these items,
please refer to the tutorial in our dataset web page.
