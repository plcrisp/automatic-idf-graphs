
from datetime import datetime, timedelta
from unidecode import unidecode
from dotenv import load_dotenv
from importlib import resources

from ..processing import aggregate_to_csv
from ..reader import process_data, DataSource

import os
import questionary
import pandas as pd
import requests


def load_inmet_station_parameters() -> pd.DataFrame:
    """
    Loads the INMET station parameter data from the package resources.

    This function safely accesses the 'inmet_stations.csv' file that is
    bundled with the installed package.

    Returns:
        pd.DataFrame: A DataFrame containing the station parameters.
    """
    try:
        with resources.files('idf_analysis.resources').joinpath('inmet_stations.csv').open('r', encoding='utf-8') as f:
            df = pd.read_csv(
                f,
                sep=';',
                on_bad_lines='warn' 
            )
        return df
    except FileNotFoundError:
        print("Error: Could not find the station parameters file within the package.")
        return pd.DataFrame()



def get_inmet_data(
    process: bool = True,
):
    """Baixa dados diários de precipitação de uma estação *operante* do INMET.

    Etapas realizadas:
        1. Interface interativa (CLI) para escolha de **estado**, **estação** e
           **intervalo de datas** (≤ 1 ano por chamada e ≥ data de início da estação).
        2. Chamada autenticada à API do INMET (token ``INMET_KEY`` no ``.env``).
        3. Armazenamento em
           ``./datasets/INMET_{NOME_DA_ESTACAO}/inmet_{nome_da_estacao}.csv``.
           Quando o arquivo já existe, combina dados novos e antigos removendo
           duplicatas.
        4. (Opcional) Processa o arquivo via :pyfunc:`process_data` e devolve o
           :class:`pandas.DataFrame` resultante.

    Args:
        stations_path: Caminho para o CSV de estações disponibilizado pelo INMET.
        process: Se ``True``, processa os dados após o download.

    Returns:
        pandas.DataFrame | None
            O *DataFrame* processado (quando ``process=True``) ou ``None``.
    """

    # Preparação
    load_dotenv()

    df_all_stations = load_inmet_station_parameters()

    df = (
        df_all_stations.query("CD_SITUACAO.str.lower() == 'operante'")
        .assign(
            VL_LATITUDE=lambda d: d["VL_LATITUDE"].str.replace(",", ".").astype(float),
            VL_LONGITUDE=lambda d: d["VL_LONGITUDE"].str.replace(",", ".").astype(float),
            DT_INICIO_OPERACAO=lambda d: pd.to_datetime(
                d["DT_INICIO_OPERACAO"], dayfirst=True
            ),
        )
    )

    # Seleção de estado
    estado = questionary.select(
        "Escolha o estado:", choices=sorted(df["SG_ESTADO"].unique())
    ).ask()

    # Seleção de estação
    df_estado = df.query("SG_ESTADO == @estado")
    estacao_str = questionary.select(
        "Escolha a estação:",
        choices=[f"{row.DC_NOME} ({row.CD_ESTACAO})" for _, row in df_estado.iterrows()],
    ).ask()

    cod_estacao = estacao_str.split("(")[-1].rstrip(")")
    nome_estacao = estacao_str.split(" (")[0]
    data_inicio_operacao = pd.to_datetime(
        df_estado.loc[df_estado["CD_ESTACAO"] == cod_estacao, "DT_INICIO_OPERACAO"].iat[0]
    ).to_pydatetime()

    # Entrada de datas
    def pedir_data(msg: str, min_date: datetime) -> datetime:
        """Solicita uma data ≥ *min_date* no formato DD/MM/AAAA."""
        print(f"\n📅 Data mínima: {min_date:%d/%m/%Y}")
        while True:
            try:
                raw = questionary.text(f"{msg} [DD/MM/AAAA]").ask()
                data = datetime.strptime(raw, "%d/%m/%Y")
                if data < min_date:
                    print(f"❌ A data deve ser ≥ {min_date:%d/%m/%Y}")
                    continue
                return data
            except ValueError:
                print("❌ Formato inválido. Use DD/MM/AAAA.")

    data_inicial = pedir_data("Data inicial", data_inicio_operacao)

    while True:
        data_maxima = min(data_inicial + timedelta(days=365), datetime.today())
        print(f"\n📅 Data máxima: {data_maxima:%d/%m/%Y}")
        data_final = pedir_data("Data final", data_inicial)

        if data_final > data_maxima:
            print(
                "❌ Intervalo além de 1 ano ou futuro. "
                "Para períodos maiores, execute novamente (os dados serão agregados)."
            )
        else:
            break

    # Requisição à API
    token = os.getenv("INMET_KEY")
    if not token:
        print("❌ Variável de ambiente INMET_KEY não encontrada.")
        return

    url = (
        "https://apitempo.inmet.gov.br/token/estacao/"
        f"{data_inicial.date()}/{data_final.date()}/{cod_estacao}/{token}"
    )
    print("\n📡 Consultando API…\n")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"❌ Erro {response.status_code} ao consultar API.")
        return

    dados_json = response.json()
    if not dados_json:
        print("⚠️ Nenhum dado retornado.")
        return

    # Transformação
    df_dados = (
        pd.DataFrame(dados_json)[["DT_MEDICAO", "HR_MEDICAO", "CHUVA"]]
        .rename(
            columns={
                "DT_MEDICAO": "Data Medicao",
                "HR_MEDICAO": "Hora Medicao",
                "CHUVA": "PRECIPITACAO TOTAL, DIARIO(mm)",
            }
        )
    )
    df_dados["Data Medicao"] = pd.to_datetime(df_dados["Data Medicao"])
    df_dados = df_dados.sort_values("Data Medicao")
    df_dados["Data Medicao"] = df_dados["Data Medicao"].dt.strftime("%Y-%m-%d")
    if "" not in df_dados.columns:
        df_dados[""] = ""

    # Persistência
    nome_limpo = unidecode(nome_estacao.lower().replace(" ", "_"))
    pasta = f"./datasets/INMET_{nome_estacao.upper().replace(' ', '_')}"
    os.makedirs(pasta, exist_ok=True)
    caminho_csv = os.path.join(pasta, f"inmet_{nome_limpo}.csv")

    if os.path.exists(caminho_csv):
        
        def padronizar_hora(hora):
            try:
                # Se já estiver no formato HH:MM, apenas retorne
                if isinstance(hora, str) and ":" in hora:
                    return hora

                # Se for numérico, converte para string no formato HH:MM
                hora_str = str(int(float(hora))).zfill(4)
                return f"{hora_str[:2]}:{hora_str[2:]}"
            except Exception:
                return "00:00"  # fallback seguro
        
        df_existente = (
            pd.read_csv(caminho_csv, sep=";")
            .loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
        )

        df_existente["DataHora"] = pd.to_datetime(
            df_existente["Data Medicao"] + " " + df_existente["Hora Medicao"].apply(padronizar_hora),
            format="%Y-%m-%d %H:%M"
        )
        df_dados["DataHora"] = pd.to_datetime(
            df_dados["Data Medicao"] + " " + df_dados["Hora Medicao"].apply(padronizar_hora),
            format="%Y-%m-%d %H:%M"
        )       

        df_dados = (
            pd.concat([df_existente, df_dados], ignore_index=True)
            .drop_duplicates("DataHora", keep="last")
            .sort_values("DataHora")
        )
        
        df_dados["Data Medicao"] = df_dados["DataHora"].dt.strftime("%Y-%m-%d")

        # Remover coluna auxiliar
        df_dados.drop(columns=["DataHora"], inplace=True)

    df_dados.to_csv(caminho_csv, index=False, sep=";")
    print(f"\n✅ Dados salvos em: {caminho_csv}")
    
    # Pós‑processamento
    if process:
        print()
        df = process_data(
            source=DataSource.INMET,
            data_path=caminho_csv,
        )
        
        aggregate_to_csv(
            df=df,
            name= 'inmet_' + nome_limpo,
            directory='./results/inmet_' + nome_limpo
        )
        
        return df
