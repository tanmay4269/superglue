import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch

from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, frame2tensor)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)

superglue = 'outdoor'
max_keypoints = 2048
keypoint_threshold = 0.05
nms_radius = 5
sinkhorn_iterations = 20
match_threshold = 0.9

config = {
    'superpoint': {
        'nms_radius': nms_radius,
        'keypoint_threshold': keypoint_threshold,
        'max_keypoints': max_keypoints
    },
    'superglue': {
        'weights': superglue,
        'sinkhorn_iterations': sinkhorn_iterations,
        'match_threshold': match_threshold,
    }
}

matching = Matching(config).eval().to(device)

def generate_pano(image0, image1):  
  inp0 = frame2tensor(image0, device)
  inp1 = frame2tensor(image1, device)
  
  # Perform the matching.
  pred = matching({'image0': inp0, 'image1': inp1})
  pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
  kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
  matches, conf = pred['matches0'], pred['matching_scores0']
  
  valid = matches > -1
  point_set1 = kpts0[valid]
  point_set2 = kpts1[matches[valid]]

  #find Homography between two source images
  H, status = cv.findHomography(point_set1, point_set2, cv.RANSAC, 5.0) 

  # Applies a homogeneous transformation to an image.
  # To transform the right image to left we need to consider the inverse.
  panorama = cv.warpPerspective(image1, np.linalg.inv(H), (960, 960)) 
  panorama[0:image0.shape[0], 0:image0.shape[1]] = image0

  return panorama
