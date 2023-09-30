from ultralytics import YOLO
import cv2
import argparse

import torch

torch.cuda.set_device(0)

def run(source="source.mp4"):

    yolov8_model_path = YOLO('models/yolov8n-seg.pt')
    
    videocapture = cv2.VideoCapture(source)

    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success:
            break
        
        
        results = yolov8_model_path.predict(frame, save=False, classes=[0,1], device='gpu')
        annotated_frame = results[0].plot()
        cv2.imshow("Yolo", annotated_frame)

        #exit if q is press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Break the while loop

    videocapture.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type = str, default=0, help = 'video file path, leave blank if you want to use the webcam')
    return parser.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)    
