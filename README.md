# Two-View-Stereo
This project harnesses classical computer vision techniques to transform multiple 2D viewpoints into a 3D representation of a scene.

## Core Tasks:
### Rectify Two Views:
- **Camera Configuration**: Understand the camera setup and derive necessary transformations.
- **Rotation Matrix**: Compute the rectification rotation matrix.
- **Image Rectification**: Utilize OpenCV for image rectification with the provided transformations.

### Compute Disparity Map:
- **Patch Formation**: Convert images into patch format.
- **Matching Kernels**: Implement SSD, SAD, and ZNCC matching techniques.
- **Consistency**: Implement left-right consistency for patch matching.
- **Disparity Construction**: Formulate the full disparity map based on the above steps.

### Depth Map and Point Cloud:
- **Depth Derivation**: Using the disparity map, derive a depth map.
- **Point Cloud Formation**: Back-project to form a point cloud in the camera frame.

### Post-Processing:
- **Transformation**: Transition the point cloud from the camera frame to the world frame.
- **Visualization**: Utilize the K3D library for visual representation. Ensure to capture screenshots of reconstructions using SSD, SAD, and ZNCC kernels.

### Multi-Pair Aggregation:
- **Aggregation**: The `two view` function merges results from several view pairs.
- **Final Reconstruction**: The culmination is the 3D reconstruction of a temple. This step might approximately take 10 minutes on conventional laptops.
