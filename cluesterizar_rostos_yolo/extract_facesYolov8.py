import argparse
import torch
import os
from choose_one_bbox import choose_bbox
import cv2

# I change the commentars to english bc i want to post this on gh
def extract_face(frame, yolo_model):

    # Check if we can use the gpu, otherwise, use the cpu
    device = 0 if torch.cuda.is_available() else "cpu"

    # floating pointing betwent 0 and 1 to define how much will be increased on which side
    # this could be a arg to, think about change it 
    EXPAND_RATIO = 0.01

    # Array to save the faces files
    output_images = []

    # get the predict result from yolo
    results = yolo_model.predict(frame, save=False, classes=[0,1], device=device)
    
    # get the bbox for which face detect
    for result in results:
                
        #Get the bbox coordinates in xyxy format
        bbox = result.boxes.xyxy
        
        #Get the current original images height and width
        h, w, c = result.orig_img.shape

        #Choose the most desired bbox in case there are multiple detections
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = choose_bbox(bbox,
                                                        image_size=(w,h),
                                                        center_coordinates=(0.5,0.5),
                                                        input_xyxy=True,
                                                        input_normalized=False,
                                                        return_xyxy=True,
                                                        AREA_RATIO_THRESHOLD=0.8
                                                        )
        
        #Check if there exists a valid detected bbox
        if((bbox_x1, bbox_y1, bbox_x2, bbox_y2) == (-1,-1,-1,-1)):
            continue
        
        #Get the width and height for image to be cropped
        cropped_image_w = bbox_x2 - bbox_x1
        cropped_image_h = bbox_y2 - bbox_y1
        
        output_images.append([bbox_x1, bbox_y1, cropped_image_w, cropped_image_h])

    return output_images