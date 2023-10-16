import pika
import cv2
import numpy as np

host = 'localhost'
queue_name_segmentacao = 'segmentacao'
queue_name_classificacao = 'classificacao'

# Inicialize a conexão com o servidor RabbitMQ
connection_segmentacao = pika.BlockingConnection(pika.ConnectionParameters(host = host))
channel_segmentacao = connection_segmentacao.channel()
channel_segmentacao.queue_declare(queue=queue_name_segmentacao)

# Inicialize o segmentador de faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicialize a conexão para enviar para a Máquina 3
connection_classificacao = pika.BlockingConnection(pika.ConnectionParameters(host = host))
channel_classificacao = connection_classificacao.channel()
channel_classificacao.queue_declare(queue=queue_name_classificacao)

while True:
    method_frame, header_frame, frame_bytes = channel_segmentacao.basic_get(queue = queue_name_segmentacao)

    if frame_bytes:
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)

        # Segmentação de faces
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_roi_image = frame[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi_image, (160, 160))

            face_frame_bytes = cv2.imencode('.jpg', face_roi)[1].tobytes()

            # Envie face_roi para a Máquina 3
            channel_classificacao.basic_publish(exchange='', 
                                                routing_key='face_roi', 
                                                body=face_frame_bytes)

        cv2.imshow('ROI', face_roi_image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

connection_segmentacao.close()
connection_classificacao.close()
