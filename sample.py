import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# Root directory of the project
ROOT_DIR = os.path.abspath("/data/coq18yj/Mask_RCNN/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from mrcnn.pose_estimation import *
from mrcnn.icp2d import icp2D
from mrcnn.MeshPly import MeshPly

from samples.cat import cat

import scipy
import skimage

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    fig.tight_layout()
    return ax
def get_3D_corners(vertices):
    
    min_x = np.min(vertices[0,:])
    max_x = np.max(vertices[0,:])
    min_y = np.min(vertices[1,:])
    max_y = np.max(vertices[1,:])
    min_z = np.min(vertices[2,:])
    max_z = np.max(vertices[2,:])

    corners = np.array([[max_x, min_z, max_y],
                        [min_x, min_z, max_y],
                        [min_x, max_z, max_y],
                        [max_x, max_z, max_y],
                        [max_x, min_z, min_y],
                        [min_x, min_z, min_y],
                        [min_x, max_z, min_y],
                        [max_x, max_z, min_y]])

    return corners

# Directory to save logs and trained model
LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Inference Configuration
config = cat.ApeInferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "gpu:0"  # /cpu:0 or /gpu:0
TEST_MODE = "inference"

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                            model_dir=LOGS_DIR,
                            config=config)
# Load weights
weights_path = "/data/coq18yj/Mask_RCNN/logs/cat20200913T1804/mask_rcnn_cat_0090.h5"
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)


# Load Input
images = skimage.io.imread('/data/coq18yj/Mask_RCNN/samples/cat/dataset/val/000916/images/000916.png')
r = model.detect([images], verbose=0)[0]

# Output From Neural Network
y1, x1, y2, x2 = r['3d_projection'][0]
quaternion_xyzw = r['quaternion'][0]
mask = r['masks'][:, :,0]

# Load Object Model
mesh = MeshPly('/data/coq18yj/Mask_RCNN/models/obj_000006_cat.ply')
vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
corners = get_3D_corners(vertices)

# Camera Intrinsic Matric
intrinsic_matric_K = np.array([
        [572.4, 0, 325],
        [0, 573.5, 243.0],
        [0, 0, 1]
    ])

# Convert quaternion to rotation matrix
cam_pose_metrix = []
obj_corners = []
rotation = R.from_quat([quaternion_xyzw[0],quaternion_xyzw[1],quaternion_xyzw[2],quaternion_xyzw[3]])
pose_transform = rotation.as_matrix()
# Calculate Object Pose
new_pose_metrix = pose_metrix(intrinsic_matric_K,corners,pose_transform,images,top_left=[x1, y1],bottom_right=[x2, y2])
# Refinement Process
# new_pose_metrix = icp2D(new_pose_metrix, intrinsic_matric_K, vertices, mask = mask)

# Draw 3D Bbox
corners  = np.r_[np.array(corners.T), np.ones((1,8))].transpose()
cam_pose_metrix.append(new_pose_metrix)
obj_corners.append(corners)
visualize.display_3Dbbox(images, intrinsic_matric_K, cam_pose_metrix, obj_corners, r['class_ids'], colors = [(1.0, 0.0, 0.0)])

plt.show()   