# python -m pip install git+https://github.com/pytube/pytube
# para a versão mais recente, já que a versão do pip está desatualizada/não pega
from pytube import YouTube

# Insira o URL do vídeo que você deseja baixar
url = "https://youtu.be/bs5BfDnufdg?si=ZtnytBj0ApFEPbGM"

try:
    # Cria um objeto YouTube com o URL
    yt = YouTube(url)
    
    # Escolha a resolução e o tipo de arquivo que você deseja baixar
    # (A primeira entrada na lista é a de maior qualidade)
    # stream = yt.streams.get_highest_resolution()

    #imprime a resolução das streams
    for stream in yt.streams:
        print(stream)
    
    # Baixe o vídeo para o diretório atual
    # stream.download()
    
    print("Download concluído com sucesso!")

except Exception as e:
    print("Ocorreu um erro:", str(e))
