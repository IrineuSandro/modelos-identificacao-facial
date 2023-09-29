from ultralytics import YOLO
import cv2
import argparse
import os
import torch
from choose_one_bbox import choose_bbox

device = 0 if torch.cuda.is_available() else "cpu"

def run(source="source.mp4",
        save: bool = True,
        IMAGES_PATH = "path",
        EXPAND_RATIO: float = 0.00):
    # check if the video is working

    videocapture = cv2.VideoCapture(source)

        
    # set the yolo model (i've set a model but i'll do this in a more dynamic way)   
    yolov8_model_path = YOLO('models/yolov8n-seg.pt')
    
    # get the predict result
    frames = yolov8_model_path.predict(frame, save=False, classes=[0], device='device')

    #get the weights names based on yolo pre-trained
    weights_name = yolov8_model_path[:-3]
    # set a list to return imagens 
    output_images = []
    #If save is True and output folder doesn't exists, create one.
    if save and (not OUTPUT_PATH):
        OUTPUT_PATH =  IMAGES_PATH.split("/")[-1] + "__" + weights_name + "__output"
        if not os.path.exists(OUTPUT_PATH):
            os.mkdir(OUTPUT_PATH)
            
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        for frame in frames:

            #Get the image name
            image_name = frame.path.slit("\\")[-1]
            # Get the bbox coordinates in xyxy format
            bbox = frame.boxes.xyxy
            # Get the current images height and width
            h, w, c = frame.orig_img.shape
            # Choose the most desired bbox in case there are multiple detections
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
            output_image = cv2.cvtColor(frame.orig_img[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            

            
            #Check if save option is True
            if save: 
                
                #Get the save path
                save_name = os.path.join(OUTPUT_PATH, image_name)
                
                #Save the cropped image
                cv2.imwrite(save_name, output_image)
                
            
            #Append the output image to the return list
            output_images.append(output_image.astype)
            #exit if q is press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Break the while loop

    videocapture.release()
    cv2.destroyAllWindows()

def parse_opt():
    BASE_FOLDER = ""
    IMAGES_PATH = os.path.join(BASE_FOLDER + "test_data")
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type = str, default = 0, help = 'video file path, leave blank if you want to use the webcam')
    parser.add_argument("--save", type = bool, default = False, help ="save the image")
    parser.add_argument("--patch", type=str, default = IMAGES_PATH)
    return parser.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)    