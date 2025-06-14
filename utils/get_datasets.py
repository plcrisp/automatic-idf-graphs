"""
Este script processa dados meteorológicos de diferentes fontes (CEMADEN, INMET).
Ele carrega arquivos CSV, padroniza colunas, converte tipos de dados e organiza as informações em DataFrames.
Dependendo da fonte selecionada, a função process_data retorna os dados formatados adequadamente para análise.
A manipulação dos dados é feita com pandas, e a gestão dos arquivos utiliza pathlib.
"""



from enum import Enum
from pathlib import Path
from collections import Counter
from datetime import datetime, timedelta
from unidecode import unidecode
from dotenv import load_dotenv

from .data_processing import aggregate_to_csv


import folium
import os
import webbrowser
import questionary
import pandas as pd
import requests






class DataSource(Enum):
    """Enum para as fontes de dados meteorológicos."""
    
    CEMADEN = 'CEMADEN'
    INMET = 'INMET'
    INMET_DAILY = 'INMET_DAILY'

def convert_to_numeric(df, columns):
    """
    Converte colunas especificadas de um DataFrame para tipo numérico.

    Parâmetros:
        df (DataFrame): O DataFrame a ser processado.
        columns (list): Lista com os nomes das colunas a serem convertidas.
    
    Retorna:
        DataFrame: O DataFrame com as colunas convertidas.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df



def print_station_record_counts(df: pd.DataFrame, site_column: str = 'Site'):
    """
    Exibe a contagem de registros por estação, em ordem decrescente.
    """
    print("\nOcorrências por estação (em ordem decrescente):")
    for site, count in df[site_column].value_counts().items():
        print(f"- {site}: {count} registros")
        
        
        
def generate_cemaden_map(data_path, cemaden_df):
    print("Gerando mapa com as estações do CEMADEN...")

    cemaden_files = Path(data_path).glob('*.csv')
    all_rows = pd.concat(
        [pd.read_csv(file, sep=';') for file in cemaden_files],
        ignore_index=True,
        sort=False
    )

    all_rows['latitude'] = all_rows['latitude'].str.replace(',', '.', regex=False).astype(float)
    all_rows['longitude'] = all_rows['longitude'].str.replace(',', '.', regex=False).astype(float)

    counts = Counter(cemaden_df['Site'])
    unique_sites = all_rows[['nomeEstacao', 'latitude', 'longitude']].drop_duplicates()

    map_center = [unique_sites['latitude'].mean(), unique_sites['longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=11)

    # Normalização
    max_count = max(counts.values()) if counts else 1
    min_count = min(counts.values()) if counts else 0

    # Mapear faixas de intensidade para cores nomeadas suportadas pelo folium
    def get_icon_color(intensity):
        if intensity > 0.8:
            return 'darkgreen'
        elif intensity > 0.6:
            return 'green'
        elif intensity > 0.4:
            return 'lightgreen'
        elif intensity > 0.2:
            return 'beige'
        else:
            return 'white'

    for _, row in unique_sites.iterrows():
        name = row['nomeEstacao']
        lat = row['latitude']
        lon = row['longitude']
        count = counts.get(name, 0)
        intensity = (count - min_count) / (max_count - min_count + 1e-9)
        icon_color = get_icon_color(intensity)

        popup_text = f"{name}<br>Registros: {count}"
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color=icon_color, icon=' ')
        ).add_to(folium_map)

    os.makedirs('./results/maps', exist_ok=True)
    map_path = './results/maps/mapa_estacoes_cemaden.html'
    folium_map.save(map_path)
    print(f"Mapa salvo em {map_path}")
    webbrowser.open('file://' + os.path.realpath(map_path))
        


def process_data(source: DataSource, data_path: str, site_filter: str = None, show_station_counts: bool = False, generate_map: bool = False):
    """
    Processa dados meteorológicos de diferentes fontes.

    Parâmetros:
        source (DataSource): Enumeração que define as fontes válidas: 'CEMADEN', 'INMET', 'INMET_DAILY'.
        data_path (str): Caminho para a pasta onde os dados estão armazenados.
        site_filter (str, opcional): Nome da estação (Site) a ser filtrada no caso do CEMADEN.
        show_station_counts (bool, opcional): Se True, exibe a contagem de registros por estação (apenas para CEMADEN).

    Retornos:
        - Se source for 'CEMADEN':
            - Se site_filter for fornecido: Retorna um único DataFrame com dados apenas da estação informada.
            - Caso contrário: Levanta um erro informando que o parâmetro site_filter é obrigatório.
        - Se source for 'INMET' ou 'INMET_DAILY': Retorna dois DataFrames
          (DataFrame aut, DataFrame conv).

    Exemplo de uso:
        df = process_data(DataSource.CEMADEN, '../datasets', site_filter='AC Santana')
        df_inmet = process_data(DataSource.INMET, '../datasets/inmet.csv')
    """

    if source == DataSource.CEMADEN:
        if not site_filter:
            raise ValueError("Para o DataSource.CEMADEN, o parâmetro 'site_filter' é obrigatório.")

        print("🔁 Processando dados do DataSource.CEMADEN...")

        # Lê todos os arquivos CSV no diretório
        cemaden_files = Path(data_path).glob('*.csv')

        # Lê e concatena os arquivos CSV em um único DataFrame
        CEMADEN_df = pd.concat(
            [pd.read_csv(file, sep=';') for file in cemaden_files],
            ignore_index=True,
            sort=False
        )

        # Seleciona e renomeia colunas relevantes
        CEMADEN_df = CEMADEN_df[['nomeEstacao', 'datahora', 'valorMedida']]
        CEMADEN_df.columns = ['Site', 'Date', 'Precipitation']

        # Converte valores para o formato correto
        CEMADEN_df['Precipitation'] = CEMADEN_df['Precipitation'].str.replace(',', '.', regex=False)

        # Converte a coluna de data para datetime
        CEMADEN_df['Date'] = pd.to_datetime(CEMADEN_df['Date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

        # Remove linhas com datas inválidas
        CEMADEN_df = CEMADEN_df.dropna(subset=['Date'])

        # Extrai partes da data
        CEMADEN_df['Year'] = CEMADEN_df['Date'].dt.year
        CEMADEN_df['Month'] = CEMADEN_df['Date'].dt.month
        CEMADEN_df['Day'] = CEMADEN_df['Date'].dt.day
        CEMADEN_df['Hour'] = CEMADEN_df['Date'].dt.hour

        # Converte para tipos numéricos
        CEMADEN_df = convert_to_numeric(CEMADEN_df, ['Year', 'Month', 'Day', 'Hour', 'Precipitation'])

        # Agrupa por estação, ano, mês, dia e hora e soma as precipitações
        CEMADEN_df = CEMADEN_df.groupby(['Site', 'Year', 'Month', 'Day', 'Hour'], as_index=False).agg({'Precipitation': 'sum'})
        
        if show_station_counts:
            print_station_record_counts(CEMADEN_df)
            
        if generate_map:
            generate_cemaden_map(data_path, CEMADEN_df)

        # Filtra a estação desejada
        station_df = CEMADEN_df[CEMADEN_df['Site'] == site_filter]

        if station_df.empty:
            raise ValueError(f"Nenhum dado encontrado para a estação '{site_filter}'.")
        
        print("\n✅ Processamento concluído!\n")

        return station_df

    elif source in {DataSource.INMET, DataSource.INMET_DAILY}:
        print(f"🔁 Processando dados do {source}...")

        df = pd.read_csv(data_path, sep=';')

        if source == DataSource.INMET:
            df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
            df = df[['Date', 'Hour', 'Precipitation']]
            df['Hour'] = df['Hour'].astype(float) / 100  # Converte hora para formato decimal
            df[['Year', 'Month', 'Day']] = df.Date.str.split("-", expand=True)
            
            print("\n✅ Processamento concluído!\n")

            
            return convert_to_numeric(df, ['Year', 'Month', 'Day', 'Hour'])

        if source == DataSource.INMET_DAILY:
            df.columns = ['Date', 'Precipitation', 'Null']
            df = df[['Date', 'Precipitation']]
            df[['Year', 'Month', 'Day']] = df.Date.str.split("-", expand=True)
            
            print("\n✅ Processamento concluído!\n")
            
            return convert_to_numeric(df, ['Year', 'Month', 'Day'])

    else:
        raise ValueError(f"Fonte '{source}' não suportada.")



def get_inmet_data(
    stations_path: str = "./parameters/estacoes_inmet.csv",
    process: bool = True,
):
    """Baixa dados diários de precipitação de uma estação *operante* do INMET.

    Etapas realizadas:
        1. Interface interativa (CLI) para escolha de **estado**, **estação** e
           **intervalo de datas** (≤ 1 ano por chamada e ≥ data de início da estação).
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

    df = (
        pd.read_csv(stations_path, sep=";")
        .query("CD_SITUACAO.str.lower() == 'operante'")
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
        "https://apitempo.inmet.gov.br/token/estacao/diaria/"
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
        pd.DataFrame(dados_json)[["DT_MEDICAO", "CHUVA"]]
        .rename(
            columns={
                "DT_MEDICAO": "Data Medicao",
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
        df_existente = (
            pd.read_csv(caminho_csv, sep=";")
            .loc[:, lambda d: ~d.columns.str.startswith("Unnamed")]
        )
        df_existente["Data Medicao"] = pd.to_datetime(df_existente["Data Medicao"])
        df_dados["Data Medicao"] = pd.to_datetime(df_dados["Data Medicao"])

        df_dados = (
            pd.concat([df_existente, df_dados], ignore_index=True)
            .drop_duplicates("Data Medicao", keep="last")
            .sort_values("Data Medicao")
        )
        df_dados["Data Medicao"] = df_dados["Data Medicao"].dt.strftime("%Y-%m-%d")

    df_dados.to_csv(caminho_csv, index=False, sep=";")
    print(f"\n✅ Dados salvos em: {caminho_csv}")
    
    # Pós‑processamento
    if process:
        print()
        df = process_data(
            source=DataSource.INMET_DAILY,
            data_path=caminho_csv,
        )
        
        aggregate_to_csv(
            df=df,
            name= 'inmet_' + nome_limpo,
            directory='./results/inmet_' + nome_limpo
        )
        
        return df
