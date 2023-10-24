# Two-View-Stereo
This project implements a two-view stereo algorithm to convert multiple 2D view-points into a 3D reconstruction of the scene.

## Core Tasks:
### Rectify Two Views:
- **Camera Configuration**: Understand the camera setup and derive necessary transformations.
- **Rotation Matrix**: Compute the rectification rotation matrix.
- **Image Rectification**: Utilize OpenCV for image rectification with the provided transformations.
  ![URL_OF_THE_IMAGE](https://github.com/satyajeetburla/Two-View-Stereo/blob/main/img/img1.png)


### Compute Disparity Map:
- **Patch Formation**: Convert images into patch format.
- **Matching Kernels**: Implement SSD, SAD, and ZNCC matching techniques.
- **Consistency**: Implement left-right consistency for patch matching.
- **Disparity Construction**: Formulate the full disparity map based on the above steps.
  <p align="center">
    <img src="https://github.com/satyajeetburla/Two-View-Stereo/blob/main/img/img2.PNG" width="400" />
    <img src="https://github.com/satyajeetburla/Two-View-Stereo/blob/main/img/img3.PNG" width="400" /> 
  </p>


### Depth Map and Point Cloud:
- **Depth Derivation**: Using the disparity map, derive a depth map.
- **Point Cloud Formation**: Back-project to form a point cloud in the camera frame.
<p align="center">
   <img src="https://github.com/satyajeetburla/Two-View-Stereo/blob/main/img/img4.PNG" width="400" />
  
</p>
### Post-Processing:
- **Transformation**: Transition the point cloud from the camera frame to the world frame.
- **Visualization**: Utilize the K3D library for visual representation. Ensure to capture screenshots of reconstructions using SSD, SAD, and ZNCC kernels.
  
### Multi-Pair Aggregation:
- **Aggregation**: The `two view` function merges results from several view pairs.
- **Final Reconstruction**: The culmination is the 3D reconstruction of a temple. 
### Reconstruction using SSD:
  ![https://github.com/satyajeetburla/Two-View-Stereo/blob/main/img/img4.png](https://github.com/satyajeetburla/Two-View-Stereo/blob/main/SSD%20Two%20view%20reconstruction%20results.png)

### Reconstruction using SAD:
  ![1](https://github.com/satyajeetburla/Two-View-Stereo/blob/main/SAD%20Two%20view%20reconstruction%20results.png)
### Reconstruction using ZNCC:
![](https://github.com/satyajeetburla/Two-View-Stereo/blob/main/ZNCC%20Two%20view%20reconstruction%20results.png)
### Multi-pair aggregation: full reconstructed point cloud of the temple:
We use several view pairs for two view stereo and directly aggregate the reconstructed point cloud in the
world frame.
![](https://github.com/satyajeetburla/Two-View-Stereo/blob/main/Final%20Reconstruction%20-%20using%20two_views.png)
