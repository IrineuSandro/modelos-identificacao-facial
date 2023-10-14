import pika
import numpy as np
import json

# Crie um array NumPy de exemplo
numpy_array = np.array([1, 2, 3, 4, 5])

# Converta o array NumPy em uma lista para serialização JSON
data_to_send = numpy_array.tolist()

# Configurações de conexão RabbitMQ
host = 'localhost'  # Altere para o host correto
queue_name = 'numpy_queue'

# Função para enviar o array para a fila RabbitMQ
def send_numpy_array(data):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    
    # Serializa o array para JSON
    json_data = json.dumps(data)
    
    # Publica a mensagem na fila
    channel.basic_publish(exchange='',
                          routing_key=queue_name,
                          body=json_data)
    
    print("Array NumPy enviado para a fila RabbitMQ")
    
    connection.close()

# Chama a função para enviar o array
send_numpy_array(data_to_send)
