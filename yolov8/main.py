
import argparse
from pathlib import Path 

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.yolov8 import download_yolov8s_model

from ultralytics import YOLO
from ultralytics.utils.files import increment_path

def run(source='source.mp4', view_img=True, save_img=False, exist_ok=False):
    # define the weigth that we going to use
    #i've set one model, but is easy to do this on a dinamicy way
    #yolo have a few diferents models, idk which model chose
    yolov8_model_path = 'models/yolov8n.pt'
    detection_model = AutoDetectionModel.from_pretrained(model_type='yolov8', 
                                                         model_path=yolov8_model_path,
                                                         confidence_threshold=0.3,
                                                         device='gpu')
    
    # video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    # I just set the fps of the video using the video's fps, need to change this later 
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*'mp4v')

    save_dir = increment_path(Path('videoTests') / 'exp', exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f'{Path(source).stem}.mp4'), fourcc, fps, (frame_width, frame_height))


    while videocapture.isOpened():
        # check if this video is open appropriately
        success, frame = videocapture.read()
        if not success:
            break
        success, frame = videocapture.read()
        if not success:
            break

        results = get_sliced_prediction(frame,
                                        detection_model,
                                        slice_height=512,
                                        slice_width=512,
                                        overlap_height_ratio=0.2,
                                        overlap_width_ratio=0.2)
        object_prediction_list = results.object_prediction_list

        boxes_list = []
        clss_list = []
        for ind, _ in enumerate(object_prediction_list):
            boxes = object_prediction_list[ind].bbox.minx, object_prediction_list[ind].bbox.miny, \
                object_prediction_list[ind].bbox.maxx, object_prediction_list[ind].bbox.maxy
            clss = object_prediction_list[ind].category.name
            boxes_list.append(boxes)
            clss_list.append(clss)

        for box, cls in zip(boxes_list, clss_list):
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (56, 56, 255), 2)
            label = str(cls)
            t_size = cv2.getTextSize(label, 0, fontScale=0.6, thickness=1)[0]
            cv2.rectangle(frame, (int(x1), int(y1) - t_size[1] - 3), (int(x1) + t_size[0], int(y1) + 3), (56, 56, 255),
                          -1)
            cv2.putText(frame,
                        label, (int(x1), int(y1) - 2),
                        0,
                        0.6, [255, 255, 255],
                        thickness=1,
                        lineType=cv2.LINE_AA)

        if view_img:
            cv2.imshow(Path(source).stem, frame)
        if save_img:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()





# parse the command line arguments intro the function
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type = str, default='0', help = 'video file path, leave blank if you want to use the webcam')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser.parse_args()

#make a "main function" that pass the args intro the function
def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)    