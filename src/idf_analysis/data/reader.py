from enum import Enum
from pathlib import Path
from collections import Counter
from typing import Optional

import folium
import os
import webbrowser
import pandas as pd

class DataSource(Enum):
    """Enum para as fontes de dados meteorológicos."""
    CEMADEN = 'CEMADEN'
    INMET = 'INMET'



def _to_number(
    s: pd.Series, 
    fill_value: int | float = 0.0, 
    as_integer: bool = False
) -> pd.Series:
    """
    Converte uma série para tipo numérico (float ou int), substituindo 
    valores inválidos/ausentes pelo valor de 'fill_value'.

    Parâmetros
    ----------
    s : pd.Series
        A série de entrada para conversão.
        fill_value : int | float, opcional
        Valor para preencher os dados ausentes ou inválidos (padrão é 0.0).
        as_integer : bool, opcional
        Se True, converte o resultado final para inteiro. 
        Se False (padrão), retorna como float.

    Retorna
    -------
    pd.Series
        A série convertida para o tipo numérico desejado.
    """
    numeric_series = pd.to_numeric(
        s.astype(str)
         .str.strip()
         .str.replace('null', '', case=False, regex=False)
         .str.replace(',', '.', regex=False)
         .str.replace(r'[^0-9\.\-]+', '', regex=True),
        errors='coerce'
    )
    
    filled_series = numeric_series.fillna(fill_value)
    
    if as_integer:
        return filled_series.astype(int)
    else:
        return filled_series




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

    all_rows['latitude'] = _to_number(all_rows['latitude'])
    all_rows['longitude'] = _to_number(all_rows['longitude'])

    counts = Counter(cemaden_df['Site'])
    unique_sites = all_rows[['nomeEstacao', 'latitude', 'longitude']].dropna().drop_duplicates()

    map_center = [unique_sites['latitude'].mean(), unique_sites['longitude'].mean()]
    folium_map = folium.Map(location=map_center, zoom_start=11)

    max_count = max(counts.values()) if counts else 1
    min_count = min(counts.values()) if counts else 0

    def get_icon_color(intensity):
        if intensity > 0.8: return 'darkgreen'
        elif intensity > 0.6: return 'green'
        elif intensity > 0.4: return 'lightgreen'
        elif intensity > 0.2: return 'beige'
        else: return 'white'

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


      
def process_data(source: DataSource, data_path: str, site_filter: Optional[str] = None, show_station_counts: bool = False, generate_map: bool = False):
    """
    Processa dados meteorológicos de diferentes fontes.
    """
    if source == DataSource.CEMADEN:
        if not site_filter:
            raise ValueError("Para o DataSource.CEMADEN, o parâmetro 'site_filter' é obrigatório.")

        print("🔁 Processando dados do DataSource.CEMADEN...")

        cemaden_files = Path(data_path).glob('*.csv')
        CEMADEN_df = pd.concat(
            [pd.read_csv(file, sep=';') for file in cemaden_files],
            ignore_index=True,
            sort=False
        )

        CEMADEN_df = CEMADEN_df[['nomeEstacao', 'datahora', 'valorMedida']]
        CEMADEN_df.columns = pd.Index(['Site', 'Date', 'Precipitation'])

        CEMADEN_df['Precipitation'] = _to_number(CEMADEN_df['Precipitation'])
        
        CEMADEN_df['Date'] = pd.to_datetime(CEMADEN_df['Date'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        CEMADEN_df = CEMADEN_df.dropna(subset=['Date'])

        CEMADEN_df['Year'] = CEMADEN_df['Date'].dt.year
        CEMADEN_df['Month'] = CEMADEN_df['Date'].dt.month
        CEMADEN_df['Day'] = CEMADEN_df['Date'].dt.day
        CEMADEN_df['Hour'] = CEMADEN_df['Date'].dt.hour
        
        CEMADEN_df = CEMADEN_df.groupby(['Site', 'Year', 'Month', 'Day', 'Hour'], as_index=False).agg({'Precipitation': 'sum'})
        CEMADEN_df['Precipitation'] = CEMADEN_df['Precipitation'].round(2)
        
        if show_station_counts:
            print_station_record_counts(CEMADEN_df)
            
        if generate_map:
            generate_cemaden_map(data_path, CEMADEN_df)

        if site_filter != "API":
            station_df = CEMADEN_df[CEMADEN_df['Site'] == site_filter]
            if station_df.empty:
                raise ValueError(f"Nenhum dado encontrado para a estação '{site_filter}'.")
        else:
            station_df = CEMADEN_df
        
        print("\n✅ Processamento concluído!\n")
        return station_df

    if source == DataSource.INMET:
        print("🔁 Processando dados do INMET...")

        with open(data_path, 'r', encoding='latin1', errors='ignore') as f:
            header = f.readline().strip()
        parts = [p.strip().lower() for p in header.split(';')]
        is_hourly = any('hora' in p for p in parts)

        usecols = [0, 1, 2] if is_hourly else [0, 1]
        df = pd.read_csv(
            data_path, sep=';', usecols=usecols, dtype=str,
            skipinitialspace=True, encoding='latin1'
        )

        df.columns = pd.Index(['Date', 'Hour', 'Precipitation']) if is_hourly else pd.Index(['Date', 'Precipitation'])

        df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day

        df['Precipitation'] = _to_number(df['Precipitation'])
        if is_hourly:
            df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce') / 100.0
            df['Hour'] = _to_number(df['Hour'],as_integer=True)

        print("✅ Detectado:", "INMET (horário)" if is_hourly else "INMET_DAILY (diário)")

        for c in ['Year', 'Month', 'Day'] + (['Hour'] if is_hourly else []):
            df[c] = pd.to_numeric(df[c], errors='coerce')

        order = ['Year', 'Month', 'Day'] + (['Hour'] if is_hourly else []) + ['Precipitation']
        return df[order].sort_values(order[:-1]).reset_index(drop=True)

    else:
        raise ValueError(f"Fonte '{source}' não suportada.")