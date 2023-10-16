import cv2
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pika

dataset = 'dataset_simples'
# backup_treinamento = 'faces_embeddings_done_4classes.npz'
backup_treinamento = 'teste.npz'
target_size = (160,160)

# modelo de extração de caracteristicas faciais (FaceNet)
embedder = FaceNet()

# modelo de segmentação de rostos (MTCNN)
detector = MTCNN()

# extrair rosto de uma única imagem
def extract_face(filename):
    print(filename)
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x,y,w,h = detector.detect_faces(img)[0]['box']
    x,y = abs(x), abs(y)
    face = img[y:y+h, x:x+w]
    face_arr = cv2.resize(face, target_size)
    print(face_arr.shape)
    return face_arr

# carregar as imagens de rostos, dentro da pasta de uma label
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

    np.savez_compressed(backup_treinamento, EMBEDDED_X, Y)

# itera pelo dataset, e se a label não estiver em Y, adiciona a label a Y, as imagens a X, e as embeddings a EMBEDDED_X
print('Adicionando novas classes ao modelo...')
y = Y.tolist()
x = EMBEDDED_X.tolist()

adicionou = False
for sub_dir in os.listdir(dataset):
    if sub_dir not in Y:
        adicionou = True
        print(f"Adicionando {sub_dir} ao modelo...")
        path = dataset +'/'+ sub_dir+'/'
        FACES = load_faces(path)
        labels = [sub_dir for _ in range(len(FACES))]
        print(f"{sub_dir} carregado com sucesso: {len(FACES)} rostos.")
        y.extend(labels)
        for img in FACES:
            x.append(get_embedding(img))

print(Y)

if adicionou:
    EMBEDDED_X, Y = np.asarray(x), np.asarray(y)
    print('Salvando novas classes...')
    np.savez_compressed(backup_treinamento, EMBEDDED_X, Y)

encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(EMBEDDED_X, Y, shuffle=True, random_state=17)

model = SVC(kernel='linear', probability=True)
model.fit(X_train, Y_train)

ypreds_test = model.predict(X_test)

print(accuracy_score(Y_test,ypreds_test))

host = 'localhost'
queue_name = 'classificacao'

# Inicialize a conexão com o servidor RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host = host))
channel = connection.channel()
channel.queue_declare(queue=queue_name)

while True:
    method_frame, header_frame, face_roi_bytes = channel.basic_get(queue=queue_name)

    if face_roi_bytes:
        face_roi = cv2.imdecode(np.frombuffer(face_roi_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Classificação
        label = model.predict([face_roi])
        confidence = model.predict_proba([face_roi])[0].max() * 100

        person = str(encoder.inverse_transform(label)[0])

        print(f"Previu {person} com {confidence:.2f}%")
