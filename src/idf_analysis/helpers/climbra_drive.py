# [DEPRECATED]
import os
import re
from typing import Tuple, Optional
from dotenv import load_dotenv
import pandas as pd

# --- Google Drive API (somente leitura) ---
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

load_dotenv()

DRIVE_READONLY_SCOPE = ["https://www.googleapis.com/auth/drive.readonly"]

def build_drive_service(credentials_json_path: Optional[str] = None):
    if credentials_json_path is None:
        credentials_json_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_json_path or not os.path.exists(credentials_json_path):
        raise RuntimeError(
            "Credenciais não encontradas. Defina GOOGLE_APPLICATION_CREDENTIALS "
            "com o caminho do JSON da Service Account."
        )
    creds = Credentials.from_service_account_file(credentials_json_path, scopes=DRIVE_READONLY_SCOPE)
    return build("drive", "v3", credentials=creds, cache_discovery=False)



def extract_folder_id(folder_url_or_id: str) -> str:
    m = re.search(r"/folders/([A-Za-z0-9_\-]+)", folder_url_or_id)
    if m:
        return m.group(1)
    return folder_url_or_id



def find_file_id_in_folder(service, folder_id: str, filename: str) -> str:
    q = f"'{folder_id}' in parents and name = '{filename}' and trashed = false"
    resp = service.files().list(
        q=q,
        fields="files(id,name,md5Checksum,modifiedTime,size,mimeType)",
        pageSize=10
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError(f"Arquivo '{filename}' não encontrado na pasta {folder_id}.")
    files.sort(key=lambda f: f.get("modifiedTime", ""), reverse=True)
    return files[0]["id"]



def download_file_from_drive(service, file_id: str, dest_path: str):
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()



def ensure_parquet_cached(service, file_id: str, cache_dir: str = "./.cache/climbra") -> str:
    """
    Baixa o PARQUET do Drive (se necessário) e guarda no cache.
    Retorna o caminho local do Parquet.
    """
    os.makedirs(cache_dir, exist_ok=True)
    parquet_path = os.path.join(cache_dir, f"{file_id}.parquet")

    if not os.path.exists(parquet_path):
        # Baixa diretamente o parquet bruto
        download_file_from_drive(service, file_id, parquet_path)

    return parquet_path



def scenario_to_filename(scenario: str) -> str:
    mapping = {
        "historical": "MIROC6-pr-hist-Basins.parquet",
        "ssp245":     "MIROC6-pr-ssp245-Basins.parquet",
        "ssp585":     "MIROC6-pr-ssp585-Basins.parquet",
    }
    if scenario not in mapping:
        raise ValueError(f"Cenário inválido: {scenario}")
    return mapping[scenario]



def extract_climbra_from_drive(selection_info: dict,
                               drive_folder_url_or_id: str,
                               credentials_json_path: Optional[str] = None,
                               output_dir: str = "./results",
                               cache_dir: str = "./.cache/climbra") -> Tuple[pd.DataFrame, str]:
    """
    Usa a seleção feita no get_climbra_data() para:
      - baixar (ou usar cache) o arquivo parquet bruto do Drive,
      - ler apenas as colunas necessárias,
      - filtrar pelo período,
      - salvar resultado como CSV.
    """
    scenario   = selection_info["data_request"]["scenario"]
    start_year = selection_info["data_request"]["start_year"]
    end_year   = selection_info["data_request"]["end_year"]
    col_name   = selection_info["data_request"]["column_name"]
    catch_id   = col_name.split("_", 1)[1]  # 'CABra_123' → '123'

    service   = build_drive_service(credentials_json_path)
    folder_id = extract_folder_id(drive_folder_url_or_id)
    filename  = scenario_to_filename(scenario)

    file_id       = find_file_id_in_folder(service, folder_id, filename)
    local_parquet = ensure_parquet_cached(service, file_id, cache_dir=cache_dir)

    # lê só as colunas necessárias
    df_selected = pd.read_parquet(local_parquet, columns=["year", "month", "day", col_name])

    # filtra período
    df_selected = df_selected[(df_selected["year"] >= start_year) &
                              (df_selected["year"] <= end_year)].reset_index(drop=True)

    # renomeia coluna para consistência
    df_selected = df_selected.rename(columns={"year": "Year"})
    df_selected = df_selected.rename(columns={"month": "Month"})
    df_selected = df_selected.rename(columns={"day": "Day"})
    df_selected = df_selected.rename(columns={col_name: "Precipitation"})

    # salva resultado em CSV (em vez de parquet)
    output_dir = f"{output_dir}/CABra{catch_id}/{scenario}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{start_year}-{end_year}_daily.csv")
    df_selected.to_csv(output_path, index=False)

    return df_selected, output_path