from idf_analysis.data.api.cemaden import get_cemaden_data
from idf_analysis.data.api.inmet import get_inmet_data
from idf_analysis.data.api.climbra import get_climbra_data, AllowedRootFolders

import questionary
from dotenv import load_dotenv
import os

# Load environment variables
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

def main():
    escolha = questionary.select(
        "Qual base de dados deseja baixar?",
        choices=["CEMADEN", "INMET", "CLIMBra"]
    ).ask()

    if escolha == "CEMADEN":
        get_cemaden_data()
    elif escolha == "INMET":
        get_inmet_data()
    elif escolha == "CLIMBra":
        get_climbra_data(
            allowed_roots=[AllowedRootFolders.GriddedData],
            chunk_size=1<<16,  # 64 KB para melhor throughput
            max_retries=3,
            timeout=60,  # segundos
            show_progress=True,
        )
    else:
        print("Nenhuma opção válida selecionada. Encerrando.")

if __name__ == "__main__":
    main()