from ultralytics import YOLO
import argparse
import torch
import os
from choose_one_bbox import choose_bbox
import cv2
from imageio import imwrite


# I change the commentars to english bc i want to post this on gh
def extract_face(cap: cv2.VideoCapture):

    # Check if we can use the gpu, otherwise, use the cpu
    device = 0 if torch.cuda.is_available() else "cpu"

    # floating pointing betwent 0 and 1 to define how much will be increased on which side
    # this could be a arg to, think about change it 
    EXPAND_RATIO = 0.01

    # Define the yolov8 model that we will use/ This is one pre-trained model that i find it here  https://github.com/akanametov/yolov8-face
    yolov8_model_path = YOLO('models/yolov8n-face.pt')

    # Array to save the faces files
    output_images = []

    while cap.isOpened():
        # Check if the video cap is ok
        sucess, frame = cap.read()
        if not sucess:
            break
        # get the predict result from yolo
        results = yolov8_model_path.predict(frame, save=False, classes=[0,1], device=device)
        
        # get the bbox for which face detect
        for result in results:
            
            #Extract the image name
            image_name = result.path.split("\\")[-1]
            
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
            
            #Calculate the cropped images coordinated regarding the EXPAND_RATIO
            left = int(max(bbox_x1-EXPAND_RATIO*cropped_image_w, 0))
            top = int(max(bbox_y1-EXPAND_RATIO*cropped_image_h, 0))
            right = int(min(bbox_x2+EXPAND_RATIO*cropped_image_w, w))
            bottom = int(min(bbox_y2+EXPAND_RATIO*cropped_image_h, h))
            
            #Obtain the cropped image
            target_size = (160,160) # here we define a default size of image
            output_image = [cv2.resize(image, target_size) for image in output_images]
            output_images.append(output_image)
    return output_images