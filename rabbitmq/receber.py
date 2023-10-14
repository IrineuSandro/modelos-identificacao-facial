import pika
import numpy as np
import json

# Configurações de conexão RabbitMQ
host = 'localhost'  # Altere para o host correto
queue_name = 'numpy_queue'

# Função para receber e processar o array NumPy
def receive_numpy_array(ch, method, properties, body):
    # Desserializa o JSON de volta para uma lista
    json_data = body.decode('utf-8')
    data = json.loads(json_data)
    
    # Converte a lista de volta para um array NumPy
    numpy_array = np.array(data)
    
    # Faça algo com o array NumPy, como imprimi-lo
    print("Array NumPy recebido:", numpy_array)

# Configura a conexão RabbitMQ e a função de callback
connection = pika.BlockingConnection(pika.ConnectionParameters(host=host))
channel = connection.channel()
channel.queue_declare(queue=queue_name)
channel.basic_consume(queue=queue_name, on_message_callback=receive_numpy_array, auto_ack=True)

print("Aguardando a recepção do array NumPy. Pressione CTRL+C para sair.")

# Inicia a escuta por mensagens
channel.start_consuming()
