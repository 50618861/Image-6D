from __future__ import division

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import json
from PIL import Image

import cam

ROOT_DIR = os.path.abspath("/data/coq18yj/image6D/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from image6D.utils import *
import image6D.model as modellib
from image6D.model import log
from image6D.pose_estimation import *

from image6D.meshply import MeshPly

from skimage.measure import find_contours
from matplotlib.patches import Polygon

from image6D.refinement import refine


# Import the definition of the neural network model and cuboids
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

def valid(model,intrinsic_matric_K,ground_truth):

    # Get dataloader
    testing_error_trans = 0.0
    testing_error_angle = 0.0
    testing_error_pixel = 0.0
    testing_samples = 0.0
    count = 0
    errs_2d             = []
    errs_3d             = []
    errs_trans          = []
    errs_angle          = []

    mesh = MeshPly('/data/coq18yj/Mask_RCNN/models/obj_000004_cam.ply')
    vertices  = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corners = get_3D_corners(vertices)
    diam  = calc_pts_diameter(np.array(mesh.vertices))

    with open ('error.csv','w') as file: 
        file.write("image_reference,image_id,errs_trans,errs_angle,pixel_dist \n")
        
    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_quaternion, gt_bbox_3d_projection =\
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)   
        print(dataset.image_reference(image_id),image_id)
        results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=0)
        for result in results:
            if len(result['rois']) != 0:
                y1, x1, y2, x2 = result['3d_projection'][0]
                quaternion_xyzw = result['quaternion'][0]
                mask = result['masks'][:, :, 0]

                rotation = R.from_quat([quaternion_xyzw[0],quaternion_xyzw[1],quaternion_xyzw[2],quaternion_xyzw[3]])
                pose_transform = rotation.as_matrix()

                pose_metrix_pr = pose_metrix(intrinsic_matric_K,corners,pose_transform,image,top_left=[x1, y1],bottom_right=[x2, y2])
                pose_metrix_pr = refine(pose_metrix_pr, intrinsic_matric_K, vertices, mask = mask)

                R_pr = np.asarray(pose_metrix_pr[0:3,0:3])
                t_pr = np.asarray([pose_metrix_pr[0:3,3]])

                gt = ground_truth[str(int(dataset.image_reference(image_id)))][0]
                R_gt = np.asarray(np.resize(gt['cam_R_m2c'],(3,3)))
                t_gt = np.asarray(np.resize(gt['cam_t_m2c'],(1,3)))

                # Compute translation error
                trans_dist   = np.sqrt(np.sum(np.square(t_gt - t_pr)))
                errs_trans.append(trans_dist)
                
                # Compute angle error
                angle_dist   = calcAngularDistance(R_gt, R_pr)
                errs_angle.append(angle_dist)
                
                # vertices = mesh_to_vertices[0]

                # Compute pixel error
                Rt_gt        = np.concatenate((R_gt, t_gt.T), axis=1)
                Rt_pr        = np.concatenate([R_pr, t_pr.T], axis=1)
                proj_2d_gt   = compute_projection(vertices, Rt_gt, intrinsic_matric_K)
                proj_2d_pred = compute_projection(vertices, Rt_pr, intrinsic_matric_K) 
                dis_pix = proj_2d_gt - proj_2d_pred
                # print(dis_pix)
                norm         = np.linalg.norm(dis_pix, axis=0)
                pixel_dist   = np.mean(norm)
                errs_2d.append(pixel_dist)
                

                # Compute 3D distances
                transform_3d_gt   = compute_transformation(vertices, Rt_gt) 
                transform_3d_pred = compute_transformation(vertices, Rt_pr)  
                dis_3D = transform_3d_gt - transform_3d_pred

                norm3d            = np.linalg.norm(dis_3D, axis=0)
                vertex_dist       = np.mean(norm3d)    
                errs_3d.append(vertex_dist) 
                # Sum errors
                testing_error_trans  += trans_dist
                testing_error_angle  += angle_dist
                testing_error_pixel  += pixel_dist
                testing_samples      += 1
                count = count + 1

                with open ('/data/coq18yj/image6D/samples/cam/error.csv','a') as file:
                    s = '{}, {},{:.15f},{:.15f},{:.15f}\n'.format(
                        dataset.image_reference(image_id),image_id,trans_dist,angle_dist,pixel_dist) 
                    file.write(s)

                # thewriter.writerow({'image_reference': dataset.image_reference(image_id) ,'image_id':image_id,'errs_trans': trans_dist,'errs_angle':angle_dist,'pixel_dist':pixel_dist})
            # print(errs_3d) 
    print('diam:',diam)
    print('errs_trans',errs_trans)
    print('errs_angle',errs_angle)
    print('pixel_dist',pixel_dist)
    # Compute 2D projection error, 6D pose error, 5cm5degree error

    px_threshold = 5 # 5 pixel threshold for 2D reprojection error is standard in recent sota 6D object pose estimation works 
    eps          = 1e-5
    acc          = len(np.where(np.array(errs_2d) <= px_threshold)[0]) * 100. / (len(errs_2d)+eps)
    acc3d10      = len(np.where(np.array(errs_3d) <= diam * 0.1)[0]) * 100. / (len(errs_3d)+eps)
    acc5cm5deg   = len(np.where((np.array(errs_trans) <= 50) & (np.array(errs_angle) <= 5))[0]) * 100. / (len(errs_trans)+eps)
    acc5cm       = len(np.where(np.array(errs_trans) <= 50)[0]) * 100. / (len(errs_trans)+eps)
    acc5deg      = len(np.where(np.array(errs_angle) <= 5)[0]) * 100. / (len(errs_angle)+eps)
    mean_err_2d  = np.mean(errs_2d)
    nts = float(testing_samples)
    print('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    print('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    print('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    print('   Acc using 5 cm  metric = {:.2f}%'.format(acc5cm))
    print('   Acc using 5 degree metric = {:.2f}%'.format(acc5deg))
    print("   Mean 2D pixel error is %f, Mean vertex error is %f" % (mean_err_2d, np.mean(errs_3d)))
    print('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts) )

    f  = open( 'result' +'.txt', 'w')
    f.write('   Acc using {} px 2D Projection = {:.2f}%'.format(px_threshold, acc))
    f.write('   Acc using 10% threshold - {} vx 3D Transformation = {:.2f}%'.format(diam * 0.1, acc3d10))
    f.write('   Acc using 5 cm 5 degree metric = {:.2f}%'.format(acc5cm5deg))
    f.write("   Mean 2D pixel error is %f, Mean vertex error is %f" % (mean_err_2d, np.mean(errs_3d)))
    f.write('   Translation error: %f m, angle error: %f degree, pixel error: % f pix' % (testing_error_trans/nts, testing_error_angle/nts, testing_error_pixel/nts))
    f.write("\n") 

if __name__ == "__main__":
            
    intrinsic_matric_K = np.array([
            [572.4, 0, 325],
            [0, 573.5, 243.0],
            [0, 0, 1]
        ])

    # Directory to save logs and trained model
    LOGS_DIR = os.path.join(ROOT_DIR, "logs")

    # Dataset directory
    DATASET_DIR = os.path.join(ROOT_DIR, "samples/cam/dataset")

    # Inference Configuration
    config = cam.ApeInferenceConfig()
    config.display()

    # Device to load the neural network on.
    # Useful if you're training a model on the same 
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "gpu:0"  # /cpu:0 or /gpu:0
    TEST_MODE = "inference"

    dataset = cam.ApeDataset()
    dataset.load_ape(DATASET_DIR, "val")
    dataset.prepare()
    # print(dataset())

    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference",
                                model_dir=LOGS_DIR,
                                config=config)

    weights_path = "/data/coq18yj/Mask_RCNN/logs/cam20200914T1357/mask_rcnn_cam_0090.h5"

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # print(corners)
    jsonfile = open('scene_gt_cam.json', 'r')

    ground_truth = json.load(jsonfile)

    valid(model, intrinsic_matric_K, ground_truth)







