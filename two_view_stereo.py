import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data
from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):
    """Given the rectify rotation, compute the rectified view and corrected projection matrix

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    R_irect,R_jrect : [3,3]
        p_rect_left = R_irect @ p_i
        p_rect_right = R_jrect @ p_j
    K_i,K_j : [3,3]
        original camera matrix
    u_padding,v_padding : int, optional
        padding the border to remove the blank space, by default 20

    Returns
    -------
    [H,W,3],[H,W,3],[3,3],[3,3]
        the rectified images
        the corrected camera projection matrix. WE HELP YOU TO COMPUTE K, YOU DON'T NEED TO CHANGE THIS
    """
    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    """Student Code Starts"""
    K_i_inv = np.linalg.inv(K_i)
    K_j_inv = np.linalg.inv(K_j)
    H_i = K_i_corr @ R_irect @ K_i_inv
    H_j = K_j_corr @ R_jrect @ K_j_inv
    #print(H_i.shape)
    #print(H_j.shape)
    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, (w_max, h_max))
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, (w_max, h_max))

    """Student Code Ends"""

    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw):
    """Compute the transformation that transform the coordinate from j coordinate to i

    Parameters
    ----------
    R_iw, R_jw : [3,3]
    T_iw, T_jw : [3,1]
        p_i = R_iw @ p_w + T_iw
        p_j = R_jw @ p_w + T_jw
    Returns
    -------
    [3,3], [3,1], float
        p_i = R_ij @ p_j + T_ij, B is the baseline
    """

    """Student Code Starts"""
    R_ij = (R_iw @ R_jw.T)
    T_ij = (-R_iw @( R_jw.T @ T_jw) )+ T_iw
    B = np.linalg.norm(T_ij)

    """Student Code Ends"""

    return R_ij, T_ij, B


def compute_rectification_R(T_ij):
    """Compute the rectification Rotation

    Parameters
    ----------
    T_ij : [3,1]

    Returns
    -------
    [3,3]
        p_rect = R_irect @ p_i
    """
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ij.squeeze(-1) / (T_ij.squeeze(-1)[1] + EPS)

    """Student Code Starts"""
    a = np.array([0, 0, 1])
    e_i = e_i / np.linalg.norm(e_i)
    cr = np.cross(e_i.flatten(), a.flatten())
    cr_normalized = cr / np.linalg.norm(cr)
    x = np.cross(cr_normalized.flatten(), e_i.flatten())
    x = x / np.linalg.norm(x)
    R_irect = np.vstack((cr_normalized, e_i, x))
    """Student Code Ends"""

    return R_irect


def ssd_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    ssd = np.zeros((src.shape[0], dst.shape[0]))
    for i in range(src.shape[0]):
        patch = src[i,:,:]
        error = (dst - patch)**2
        f_sum = np.sum(np.sum(error, axis = -1), axis = -1)
        ssd[i,:] = f_sum
    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    """Compute SSD Error, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        error score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    sad = np.zeros((src.shape[0], dst.shape[0]))
    for i in range(src.shape[0]):
        patch= src[i,:,:]
        error = abs(dst - patch)
        f_sum = np.sum(np.sum(error, axis = -1), axis = -1)
        sad[i,:] = f_sum
    """Student Code Ends"""

    return sad  # M,N


def zncc_kernel(src, dst):
    """Compute negative zncc similarity, the RGB channels should be treated saperately and finally summed up

    Parameters
    ----------
    src : [M,K*K,3]
        M left view patches
    dst : [N,K*K,3]
        N right view patches

    Returns
    -------
    [M,N]
        score for each left patches with all right patches.
    """
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    zncc = np.zeros((src.shape[0], dst.shape[0]))
    for i in range(src.shape[0]):
        #Left
        patch_l = src[i, :, :]
        sigma = np.std(patch_l, axis=0).reshape((1, src.shape[2]))
        #For Safe division
        sigma[np.where(sigma == 0)] = EPS
        patch_mean = np.mean(patch_l, axis=0)
        patch_left_mean = patch_l - patch_mean
        #Right
        patch_r_std = np.std(dst, axis=1)
        sigma_new = (patch_r_std * sigma).reshape((dst.shape[0], 1, dst.shape[2]))
        #For Safe division
        sigma_new[np.where(sigma_new == 0)] = EPS
        patch_r = np.mean(dst, axis=1).reshape((dst.shape[0], 1, dst.shape[2]))
        patch_r_mean = dst - patch_r

        #ZNCC
        zncc[i, :]  = ((patch_r_mean * patch_left_mean) / sigma_new).sum(axis=-1).sum(axis=-1)

    """Student Code Ends"""

    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):
    """get patch buffer for each pixel location from an input image; For boundary locations, use zero padding

    Parameters
    ----------
    image : [H,W,3]
    k_size : int, must be odd number; your function should work when k_size = 1

    Returns
    -------
    [H,W,k_size**2,3]
        The patch buffer for each pixel
    """

    """Student Code Starts"""
    pad_size = k_size // 2
    H = image.shape[0]
    W = image.shape[1]
    D = image.shape[2]
    patch_b = np.zeros((H, W, k_size ** 2, D))

    for i in range(D):
        padded_image = np.pad(image[:, :, i], pad_size)
        for x in range(H):
            for y in range(W):
                padded_row, padded_column = x + pad_size, y + pad_size
                #max() to ensure that the sub-image does not exceed the boundaries of the padded image
                patch_b[x, y, :, i] = padded_image[max(0, padded_row - pad_size):padded_row + pad_size + 1, max(0, padded_column - pad_size):padded_column + pad_size + 1].flatten()
    patch_buffer = patch_b
    """Student Code Starts"""

    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):
    """Compute the disparity map from two rectified view

    Parameters
    ----------
    rgb_i,rgb_j : [H,W,3]
    d0 : see the hand out, the bias term of the disparty caused by different K matrix
    k_size : int, optional
        The patch size, by default 3
    kernel_func : function, optional
        the kernel used to compute the patch similarity, by default ssd_kernel
    img2patch_func : function, optional
        this is for auto-grader purpose, in grading, we will use our correct implementation 
        of the image2path function to exclude double count for errors in image2patch function

    Returns
    -------
    disp_map: [H,W], dtype=np.float64
        The disparity map, the disparity is defined in the handout as d0 + vL - vR

    lr_consistency_mask: [H,W], dtype=np.float64
        For each pixel, 1.0 if LR consistent, otherwise 0.0
    """

    """Student Code Starts"""
    patches_left = img2patch_func(rgb_i.astype(float) / 255.0, k_size)
    patches_right = img2patch_func(rgb_j.astype(float) / 255.0, k_size)
    H, W = rgb_i.shape[:2]
    vi_idx, vj_idx = np.arange(H), np.arange(H)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    disp_map = np.zeros((H, W), dtype=np.float64)
    lr_consistency_mask = np.zeros((H, W), dtype=np.float64)

    for i in range(W):
        buf_left, buf_right = patches_left[:, i], patches_right[:, i]
        # Compute patch similarity.
        patch_similarity = kernel_func(buf_left, buf_right)
        # Compute disparity for each row in the current column.
        for j in range(patch_similarity.shape[0]):
            # Compute the column index of the best matched patch in the right image.
            best_matched_right_patch = np.argmin(patch_similarity[j, :])
            # Compute the disparity for the current pixel.
            disp_map[j, i] = disp_candidates[j, best_matched_right_patch]
            # Check if the best matched patch in the left image corresponds to the current pixel.
            best_matched_left_patch = patch_similarity[:, best_matched_right_patch].argmin()
            is_lr_consistent = int(best_matched_left_patch == j)
            lr_consistency_mask[j, i] = is_lr_consistent
    """Student Code Ends"""

    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    """Given disparity map d = d0 + vL - vR, the baseline and the camera matrix K
    compute the depth map and backprojected point cloud

    Parameters
    ----------
    disp_map : [H,W]
        disparity map
    B : float
        baseline
    K : [3,3]
        camera matrix

    Returns
    -------
    [H,W]
        dep_map
    [H,W,3]
        each pixel is the xyz coordinate of the back projected point cloud in camera frame
    """

    """Student Code Starts"""
    # Extract the camera parameters
    img_c_x = K[0, 2]
    img_cent_y = K[1, 2]
    a = B * K[1, 1]
    d_invert = 1.0 / disp_map
    dep_map = a * d_invert

    xyz_cam = np.zeros((disp_map.shape[0], disp_map.shape[1], 3))
    x, y = np.meshgrid(np.arange(disp_map.shape[1]), np.arange(disp_map.shape[0]))
    xyz_cam[:, :, 0] = (x - img_c_x) * dep_map / K[0, 0]
    xyz_cam[:, :, 1] = (y - img_cent_y) * dep_map / K[1, 1]
    xyz_cam[:, :, 2] = dep_map

    # for i in range(disp_map.shape[0]):
    #     for j in range(disp_map.shape[1]):
    #         depth = dep_map[i, j]
    #         y_coord = (i - img_cent_y) * depth / K[1, 1]
    #         x_coord = (j - img_c_x) * depth / K[0, 0]
    #         xyz_cam[i, j, :] = np.array([x_coord, y_coord, depth])
    """Student Code Ends"""

    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    """
    Your goal in this function is: 
    given pcl_cam [N,3], R_wc [3,3] and T_wc [3,1]
    compute the pcl_world with shape[N,3] in the world coordinate
    """

    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    """Student Code Starts"""
    #R_wc = R_cw.T
    #T_wc = -R_cw.T@T_cw
    pcl_world = (R_wc.T@ pcl_cam.T - (R_wc.T@ T_wc)).T
    """Student Code Ends"""

    # np.savetxt("./debug_pcl_world.txt", np.concatenate([pcl_world, pcl_color], -1))
    # np.savetxt("./debug_pcl_rect.txt", np.concatenate([pcl_cam, pcl_color], -1))

    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_iw, T_iw = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_jw, T_jw = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ij, T_ij, B = compute_right2left_transformation(R_iw, T_iw, R_jw, T_jw)
    assert T_ij[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ij)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ij,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_iw,
        T_wc=R_irect @ T_iw,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    #viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
