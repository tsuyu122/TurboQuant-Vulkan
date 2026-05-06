import os
from datetime import datetime

def main():
    print("Sistema de agentes funcionando!")
    
    # Lista arquivos na pasta agents/
    try:
        files = os.listdir('agents/')
        print(f"Arquivos em agents/: {files}")
    except Exception as e:
        print(f"Erro ao listar arquivos: {e}")

    # Mostra data e hora
    print(f"Data/Hora atual: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
