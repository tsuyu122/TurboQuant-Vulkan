import datetime
import os

def main():
    # 1. Imprimir mensagem de ping
    print("🏓 PING! Sistema autônomo funcionando!")

    # 2. Mostrar data/hora atual formatada
    agora = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Data/Hora: {agora}")

    # 3. Listar os 5 primeiros arquivos da pasta agents/
    print("\nPrimeiros arquivos em agents/:")
    try:
        # Tenta listar o diretório atual (agents/)
        # Como o script está dentro de agents/, listamos o diretório atual '.'
        arquivos = os.listdir('.')
        for i, arquivo in enumerate(arquivos):
            if i < 5:
                print(f"- {arquivo}")
            else:
                break
    except Exception as e:
        print(f"Erro ao listar arquivos: {e}")

    # 4. Mensagem de sucesso
    print("\n✅ Teste concluído com sucesso!")

if __name__ == "__main__":
    main()
