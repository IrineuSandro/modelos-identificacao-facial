import cv2
import os
import numpy as np

# Diretório que contém os arquivos de treinamento de cada pessoa
train_dir = 'treino/'

# Criação do classificador LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

pessoas = [f for f in os.listdir(train_dir)]

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(directory):
    image_paths = [os.path.join(directory, f) for f in os.listdir(directory)]
    face_samples = []
    ids = []

    for index, image_path in enumerate(image_paths):
        for image in [os.path.join(image_path, f) for f in os.listdir(image_path)]:

            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # Lê a imagem em escala de cinza
            
            # Detectar rosto na imagem
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                # if w < 130 or h < 130:
                #     continue
                face_roi = img[y:y+h, x:x+w]  # Selecionar a região de interesse (rosto)
                face_samples.append(face_roi)
                ids.append(index)

    return face_samples, ids

faces, ids = get_images_and_labels(train_dir)

# for face, id in zip(faces, ids):
#     cv2.imshow(f'Pessoa {id}', face)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# Treinamento do reconhecedor
recognizer.train(faces, np.array(ids))

# Inicialização da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi_gray)
        
        # Defina um limite de confiança para considerar a identificação correta
        print(label, confidence)
        if confidence < 100:
            person = pessoas[label]
        else:
            person = 'Desconhecido'
        
        cv2.putText(frame, person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('ROI Gray', roi_gray)

    cv2.imshow('Reconhecimento Facial', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
