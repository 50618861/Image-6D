"""
Pose Refinement

"""

import numpy as np
import scipy
import cv2
from scipy.spatial.transform import Rotation as R
from mrcnn import utils
from skimage.measure import find_contours
from sklearn.neighbors import NearestNeighbors
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

def fproject(rotation_matrix,tran_matrix,point,intrinsic_matric_K,contour_dilation):
    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])
    
    NEW = np.c_[rotation_matrix,tran_matrix.T]
    pose = np.r_['0,2', NEW, np.asarray([0,0,0,1])]

    projected_cuboid = intrinsic_matric_K.dot(T)
    projected_cuboid = projected_cuboid.dot(pose)

    projected_cuboid = projected_cuboid.dot(point)
    result = np.array([projected_cuboid[0]/projected_cuboid[2],projected_cuboid[1]/projected_cuboid[2]])

    test_reference = result.T

    point_project = result.T.astype(np.int64)

    mask = np.zeros(
        (480, 640), dtype=np.uint8)

    point_project[np.where(point_project[:,1]>=480),1] = 479
    point_project[np.where(point_project[:,1]<=0),1] = 1
    point_project[np.where(point_project[:,0]>=640),0] = 639
    point_project[np.where(point_project[:,0]<=0),0] = 1
    mask[point_project[:,1], point_project[:,0]] = 1

    mask_cv = np.zeros(
        (mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_cv[:,:,0],mask_cv[:,:,1],mask_cv[:,:,2] = mask,mask,mask
    kernel = np.ones((contour_dilation,contour_dilation),np.uint8)
    mask_dilation = cv2.dilate(mask_cv,kernel,iterations = 1)
    mask = mask_dilation[:,:,0]

    contours = find_contours(mask, 0)
    
    
    verts = contours[0]
    points = np.fliplr(verts)

    nbrs = NearestNeighbors(n_neighbors=1, radius=1,algorithm='kd_tree').fit(test_reference)

    control_point = point_pair(nbrs,points,test_reference)

    return control_point

def point_pair(nbrs,points,reference_points):
    distances, indices = nbrs.kneighbors(points)
    
    refer = np.squeeze(reference_points[indices])

    return refer

def refine(pose, intrinsic_matric_K, vertices, contour_dilation = 1, mask = None,e = 0.00000001, Max_Iteration = 30):
    mask_cv = np.zeros(
            (mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    mask_cv[:,:,0],mask_cv[:,:,1],mask_cv[:,:,2] = mask,mask,mask

    kernel = np.ones((contour_dilation,contour_dilation),np.uint8)
    mask_dilation = cv2.dilate(mask_cv,kernel,iterations = 1)

    mask = mask_dilation[:,:,0]
    contours = find_contours(mask, 0)
    verts = contours[0]
    reference_points = np.fliplr(verts)
   
    rotation_matrix = pose[0:3,0:3]      

    Tran = pose[0:3,3]
    nbrs = NearestNeighbors(n_neighbors=1, radius = 10,algorithm='auto').fit(reference_points)
    J = []
    for i in range(Max_Iteration):
        projection_pr_1 = fproject(rotation_matrix,Tran,vertices,intrinsic_matric_K,contour_dilation)
        refer = point_pair(nbrs,projection_pr_1,reference_points)

        J_1 = np.array(((fproject(rotation_matrix,Tran + [e,0,0],vertices,intrinsic_matric_K,contour_dilation)).flatten() - projection_pr_1.flatten())/e)
        J_2 = np.array(((fproject(rotation_matrix,Tran + [0,e,0],vertices,intrinsic_matric_K,contour_dilation)).flatten() - projection_pr_1.flatten())/e)
        J_3 = np.array(((fproject(rotation_matrix,Tran + [0,0,e],vertices,intrinsic_matric_K,contour_dilation)).flatten() - projection_pr_1.flatten())/e)


        J = [J_1, J_2, J_3]
        J = np.asarray(J)
        
        dy = refer.flatten() - projection_pr_1.flatten()
        dx = np.linalg.pinv(J.T).dot(dy.T)
        norm   = np.linalg.norm(dy, axis=0)

        Tran = Tran + dx[0:3]

        if abs(np.linalg.norm(dx)/np.linalg.norm(Tran)) <= 0.0005:
            break


    roatation_tran = np.c_[rotation_matrix,Tran.T]
    new_pose = np.r_['0,2',roatation_tran,np.asarray([0,0,0,1])]
    return new_pose

