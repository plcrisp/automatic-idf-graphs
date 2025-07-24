from idf_analysis.data.api.cemaden import get_cemaden_data
from idf_analysis.data.api.inmet import get_inmet_data

import questionary
from dotenv import load_dotenv
import os

# Load environment variables
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

def main():
    escolha = questionary.select(
        "Qual base de dados deseja baixar?",
        choices=["CEMADEN", "INMET"]
    ).ask()

    if escolha == "CEMADEN":
        get_cemaden_data()
    elif escolha == "INMET":
        get_inmet_data()
    else:
        print("Nenhuma opção válida selecionada. Encerrando.")

if __name__ == "__main__":
    main()