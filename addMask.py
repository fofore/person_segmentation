import glob
import os
import cv2
import numpy as np


original_path = "JPEGImages"
mask_path = "PNGSegmentationMask"


for ori_image_path in glob.glob(os.path.join(original_path, "*.png")):
    mask_path = original_path.replace(original_path, mask_path)
    ori_image = cv2.imread(ori_image_path)
    mask = cv2.imread(mask_path)

    pred_idx = np.where(mask == 1)

    hs, ws = ori_image.size()[0], ori_image.size()[1]
    mask_result = np.zeros((hs, ws, 3), dtype=np.uint8)
    mask_result[pred_idx[0], pred_idx[1], :] = 0, 255, 0  # 漏报率，Red

