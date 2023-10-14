import argparse
import os
import cv2
from extract_facesYolov8 import extract_face
from keras_facenet import FaceNet
import numpy as np
embedder = FaceNet()
def run (source = "source.mp4"):    
    dataset = 'dataset_simples'
    backup_treinamento = 'teste.npz'
    if os.path.exists(backup_treinamento):
        print('Carregando classes salvas antes...')
        data = np.load(backup_treinamento)
        EMBEDDED_X, Y = data['arr_0'], data['arr_1']
    else:
        x = []
        y = []

        # carregar todas as imagens de rostos, dentro de todas as pastas de labels
        for sub_dir in os.listdir(dataset):
            path = dataset +'/'+ sub_dir+'/'
            FACES = load_faces(path)
            labels = [sub_dir for _ in range(len(FACES))]
            print(f"{sub_dir} carregado com sucesso: {len(labels)} rostos.")
            x.extend(FACES)
            y.extend(labels)

        X, Y = np.asarray(x), np.asarray(y)

        EMBEDDED_X = []

        for img in X:
            EMBEDDED_X.append(get_embedding(img))

        EMBEDDED_X = np.asarray(EMBEDDED_X)

        np.savez_compressed(backup_treinamento, EMBEDDED_X, filef        

    

# parse the command line arguments intro the function
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type = str, default='0', help = 'video file path, leave blank if you want to use the webcam')
    return parser.parse_args()

def load_faces(path):
    FACES = []
    for im_name in os.listdir(path):
        try:
            filename = path + im_name
            single_face = extract_face(filename)
            FACES.append(single_face)
        except Exception as e:
            print(e)
            pass
    return FACES

def get_embedding(face_img):
    face_img = face_img.astype('float32') # 3D(160x160x3)
    face_img = np.expand_dims(face_img, axis=0) 
    # 4D (Nonex160x160x3)
    yhat= embedder.embeddings(face_img)
    return yhat[0] # 512D image (1x1x512)

#make a "main function" that pass the args intro the function
def main(opt):
    run(**vars(opt))

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)    