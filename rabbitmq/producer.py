import pika
import cv2

host = 'localhost'
queue_name = 'segmentacao'

# Inicialize a conexão com o servidor RabbitMQ
connection = pika.BlockingConnection(pika.ConnectionParameters(host = host))
channel = connection.channel()
channel.queue_declare(queue = queue_name)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Serialize o quadro (frame) e envie para a Máquina 2
    frame_bytes = cv2.imencode('.jpg', gray)[1].tobytes()
    channel.basic_publish(exchange='', 
                          routing_key=queue_name, 
                          body=frame_bytes)

    cv2.imshow('Cam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

connection.close()
