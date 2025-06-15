"""
Este script processa dados meteorológicos de diferentes fontes (CEMADEN, INMET).
Ele carrega arquivos CSV, padroniza colunas, converte tipos de dados e organiza as informações em DataFrames.
Dependendo da fonte selecionada, a função process_data retorna os dados formatados adequadamente para análise.
A manipulação dos dados é feita com pandas, e a gestão dos arquivos utiliza pathlib.
"""

from enum import Enum
from pathlib import Path
from collections import Counter

import folium
import os
import webbrowser
import pandas as pd

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

        if site_filter != "API":
            # Filtra a estação desejada
            station_df = CEMADEN_df[CEMADEN_df['Site'] == site_filter]

            if station_df.empty:
                raise ValueError(f"Nenhum dado encontrado para a estação '{site_filter}'.")
        else:
            station_df = CEMADEN_df
        
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