# open3d_interface

## Example Usage

Launch reconstruction node
`roslaunch open3d_interface yak.launch depth_image_topic:=/camera/depth_image/raw color_image_topic:=/camera/color_image/raw camera_info_topic:=/camera/camera_info`

Call service to start reconstruction
```
rosservice call /start_reconstruction "tracking_frame: 'camera'
relative_frame: 'world'
translation_distance: 0.0
rotational_distance: 0.0
live: true
tsdf_params:
  voxel_length: 0.02
  sdf_trunc: 0.04
  min_box_values: {x: 0.05, y: 0.25, z: 0.1}
  max_box_values: {x: 7.0, y: 3.0, z: 1.2}
rgbd_params: {depth_scale: 1000.0, depth_trunc: 0.75, convert_rgb_to_intensity: false}"
```

Call service to stop reconstruction
```
rosservice call /stop_reconstruction "archive_directory: '/home/ros-industrial/open3d/archive'
results_directory: '/home/ros-industrial/open3d_archive/results'"
```
