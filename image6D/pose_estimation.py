import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def Draw(img, points,color):

    thickness = 2
    cv2.line(img,tuple(points[0]), tuple(points[1]),color,thickness)
    cv2.line(img,tuple(points[1]), tuple(points[2]),color,thickness)
    cv2.line(img,tuple(points[3]), tuple(points[0]),color,thickness)
    cv2.line(img,tuple(points[3]), tuple(points[2]),color,thickness)

    # # # draw back
    cv2.line(img,tuple(points[4]), tuple(points[5]),color,thickness)
    cv2.line(img,tuple(points[5]), tuple(points[6]),color,thickness)
    cv2.line(img,tuple(points[7]), tuple(points[4]),color,thickness)
    cv2.line(img,tuple(points[7]), tuple(points[6]),color,thickness)

    # # draw sides
    cv2.line(img,tuple(points[0]), tuple(points[4]),color,thickness)
    cv2.line(img,tuple(points[7]), tuple(points[3]),color,thickness)
    cv2.line(img,tuple(points[5]), tuple(points[1]),color,thickness)
    cv2.line(img,tuple(points[2]), tuple(points[6]),color,thickness)


def pose_metrix(intrinsic_matric_K,corners,pose_transform,img,top_left,bottom_right): 

    d = np.asarray(corners)
    new_object_3D_pose = pose_transform.dot(d.T)

    length = np.max(new_object_3D_pose[0]) - np.min(new_object_3D_pose[0])
    width = np.max(new_object_3D_pose[1]) - np.min(new_object_3D_pose[1])
    high = np.max(new_object_3D_pose[2]) - np.min(new_object_3D_pose[2])

    diagonal_3D = np.sqrt(np.square(length) +np.square(high))

    diagonal_2D = np.sqrt(np.square(np.max(bottom_right)-np.max(top_left))+np.square(np.min(bottom_right)-np.min(top_left)))

    projected_cuboid_centroid =[top_left[0]+(bottom_right[0]-top_left[0])/2,top_left[1]+(bottom_right[1]-top_left[1])/2]

    Z = (diagonal_3D * intrinsic_matric_K[0,0])/(diagonal_2D*1.2) + width/2
    X = ((projected_cuboid_centroid[0] - intrinsic_matric_K[0,2])*Z)/intrinsic_matric_K[0,0] 
    Y = ((projected_cuboid_centroid[1] - intrinsic_matric_K[1,2])*Z)/intrinsic_matric_K[1,1]
    Tran = np.array([X,Y,Z])

    Zero_matrix = np.asarray([0,0,0,1])
    NEW = np.c_[pose_transform,Tran.T]
    pose = np.r_['0,2',NEW,Zero_matrix]
    return pose


def calculate_2Dpoint(intrinsic_matric_K,new_pose_metrix,corners):
    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0]
    ])

    projected_cuboid = intrinsic_matric_K.dot(T)
    projected_cuboid = projected_cuboid.dot(new_pose_metrix)
    projected_cuboid = projected_cuboid.dot(corners.T)

    result = projected_cuboid.T
    result[...,0] = result[...,0]/result[...,2]
    result[...,1] = result[...,1]/result[...,2]
    result[...,2] = result[...,2]/result[...,2]

    point = result[...,:2]
    point = np.asarray(point).astype(np.int64)
    return point

