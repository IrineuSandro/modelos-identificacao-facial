
from ultralytics import YOLO
import argparse
import torch
import os
from choose_one_bbox import choose_bbox
import cv2
from imageio import imwrite
# definir o path dos rostos
dataset_path = 'rostos/'
# faz um novo path caso nao tenha
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)


# testa usar a gpu por meio do pytourch
device = 0 if torch.cuda.is_available() else "cpu"

# float entre 0 e 1 para quantos % de cada lado sera aumentado, 0.01 = 10%
EXPAND_RATIO = 0.01
def run(source="source.mp4"):
    # definir o modelo que o yolo vai usar 
    yolov8_model_path = YOLO('models/yolov8n-face.pt')

    weights_name = "yolov8n"

    # Array para armazenar os rotos 
    output_images = []        
    
    #fazer a captura de video
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        # Verifica se deu certo ler o video 
        sucess, frame = cap.read()
        if not sucess:
            break
        # Pegar os resultados dado pelo yolo
        results = yolov8_model_path.predict(frame, save=False, classes=[0,1], device=device)
        
        # Ver rosto por rosto
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
            output_image = cv2.cvtColor(result.orig_img[top:bottom, left:right], cv2.COLOR_BGR2RGB)
            

        # Opcao para so fechar o video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  
    
    # Libere o objeto de captura e feche a janela
    cap.release()
    cv2.destroyAllWindows()            


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type = str, default=0, help = 'video file path, leave blank if you want to use the webcam')
    parser.add_argument("--save", type=bool, default=False, help= 'Choose if you want to save faces or not')
    return parser.parse_args()

def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)   