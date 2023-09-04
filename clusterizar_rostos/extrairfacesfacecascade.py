import cv2
import os

dataset_path = 'rostos/'

# Abra o vídeo
video_path = 'testeavenidapaulista.mp4'
cap = cv2.VideoCapture(video_path)

# Configure o número de frames para pular a cada N segundos
fps = cap.get(cv2.CAP_PROP_FPS)
intervalo = int(fps * 1)  # N segundos

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

id = 0
while True:
    # Pule o número de frames calculado
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + intervalo)

    # Leia o próximo frame após pular
    ret, frame = cap.read()

    # Verifique se a leitura do frame foi bem-sucedida
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        id += 1
        face = frame[y:y + h, x:x + w]

        # Salve o rosto detectado
        cv2.imwrite(dataset_path + 'face' + str(id) + '.jpg', face)

    # cv2.imshow('Frame', frame)

    # Pressione 'q' para sair do loop
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

# Libere o objeto de captura e feche a janela
cap.release()
cv2.destroyAllWindows()
