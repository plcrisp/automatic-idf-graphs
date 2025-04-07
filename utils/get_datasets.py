"""
Este script processa dados meteorológicos de diferentes fontes (CEMADEN, INMET).
Ele carrega arquivos CSV, padroniza colunas, converte tipos de dados e organiza as informações em DataFrames.
Dependendo da fonte selecionada, a função process_data retorna os dados formatados adequadamente para análise.
A manipulação dos dados é feita com pandas, e a gestão dos arquivos utiliza pathlib.
"""

import pandas as pd
from enum import Enum
from pathlib import Path


class DataSource(Enum):
    """Enum para as fontes de dados meteorológicos."""
    
    CEMADEN = 'CEMADEN'
    INMET = 'INMET'
    INMET_DAILY = 'INMET_DAILY'
    MAPLU = 'MAPLU'
    MAPLU_USP = 'MAPLU_USP'

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


def process_data(source: DataSource, data_path, year_start=None, year_end=None):
    """
    Processa dados meteorológicos de diferentes fontes.

    Parâmetros:
        source (DataSource): Enumeração que define as fontes válidas: 'CEMADEN', 'INMET', 'INMET_DAILY', 'MAPLU', 'MAPLU_USP'.
        data_path (str): Caminho para a pasta onde os dados estão armazenados.
        year_start (int, opcional): Ano inicial para filtragem, se aplicável.
        year_end (int, opcional): Ano final para filtragem, se aplicável.


    Retornos:
        - Se source for 'CEMADEN': Retorna um dicionário de DataFrames separados por Site.
        - Se source for 'INMET' ou 'INMET_DAILY': Retorna dois DataFrames
          (DataFrame aut, DataFrame conv).
        - Se source for 'MAPLU': Retorna dois DataFrames
          (DataFrame Escola, DataFrame Posto).
        - Se source for 'MAPLU_USP': Retorna um DataFrame (DataFrame USP).

    Exemplo de uso:
        df1, df2 = process_data('INMET', '../datasets')
    """

    if source == DataSource.CEMADEN:
        print("Processando dados do DataSource.CEMADEN...")

        # Lê e concatena os arquivos CSV em um único DataFrame
        # Obter todos os arquivos CSV no diretório especificado
        cemaden_files = Path(data_path).glob('CEMADEN/*.csv')

        # Lê e concatena os arquivos CSV em um único DataFrame
        CEMADEN_df = pd.concat(
            [pd.read_csv(file, sep=';') for file in cemaden_files],
            ignore_index=True,
            sort=False
        )

        # Renomeia as colunas e seleciona as relevantes
        CEMADEN_df.columns = ['1', '2', '3', '4', '5', '6', '7','8'] # a primeira coluna é desconsiderada, por isso os números estão deslocados em 1
        CEMADEN_df = CEMADEN_df[['3', '6', '7']]
        CEMADEN_df.columns = ['Site', 'Date', 'Precipitation']
        

        # Substitui vírgulas por pontos nas precipitações
        CEMADEN_df['Precipitation'] = CEMADEN_df['Precipitation'].str.replace(',', '.')

        # Divide a coluna Date em Year, Month, Day, Hour
        CEMADEN_df[['Year', 'Month', 'Day_hour']] = CEMADEN_df.Date.str.split("-", expand=True)
        CEMADEN_df[['Day', 'Hour_min']] = CEMADEN_df.Day_hour.str.split(" ", expand=True)
        CEMADEN_df[['Hour', 'Min', 'Seg']] = CEMADEN_df.Hour_min.str.split(":", expand=True)

        # Seleciona as colunas relevantes para o DataFrame final
        CEMADEN_df = CEMADEN_df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Precipitation']]
        

        # Converte as colunas especificadas para numérico
        CEMADEN_df = convert_to_numeric(CEMADEN_df, ['Year', 'Month', 'Day', 'Hour', 'Precipitation'])

        # Criar um dicionário de DataFrames separados por Site
        site_dfs = {site: CEMADEN_df[CEMADEN_df['Site'] == site] for site in CEMADEN_df['Site'].unique()}

        return site_dfs

    elif source in {DataSource.INMET, DataSource.INMET_DAILY}:
        print(f"Processando dados do {source}...")


        df = pd.read_csv(data_path, sep=';')
            
        if source == DataSource.INMET:
            df.columns = ['Date', 'Hour', 'Precipitation', 'Null']
            df = df[['Date', 'Hour', 'Precipitation']]
            df['Hour'] = df['Hour'].astype(float) / 100  # Converte hora para formato decimal
            df[['Year', 'Month', 'Day']] = df.Date.str.split("-", expand=True) 
            return convert_to_numeric(df, ['Year', 'Month', 'Day', 'Hour'])
        
        if source == DataSource.INMET_DAILY:
            df.columns = ['Date', 'Precipitation', 'Null']
            df = df[['Date', 'Precipitation']]
            df[['Year', 'Month', 'Day']] = df.Date.str.split("-", expand=True) 
            return convert_to_numeric(df, ['Year', 'Month', 'Day'])

    elif source == DataSource.MAPLU:
        print("Processando dados do DataSource.MAPLU...")

        def process_maplu_data(file_path, site_name):
            """Processa dados específicos do MAPLU."""
            df = pd.read_csv(file_path)
            df.columns = ['Site', 'Date', 'Precipitation']
            df['Site'] = site_name
            df[['Year', 'Month', 'Day_hour']] = df.Date.str.split("-", expand=True)
            df[['Day', 'Hour_min']] = df.Day_hour.str.split(" ", expand=True)
            df[['Hour', 'Min']] = df.Hour_min.str.split(":", expand=True)
            df = df[['Site', 'Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
            return convert_to_numeric(df, ['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation'])

        # Processa os dados da Escola e do Posto de Saúde
        esc_df = pd.concat(
            [process_maplu_data(f'{data_path}/MAPLU/escola{i}.csv', 'Escola Sao Bento') for i in range(year_start, year_end + 1)],
            ignore_index=True
        )
        posto_df = pd.concat(
            [process_maplu_data(f'{data_path}/MAPLU/postosaude{i}.csv', 'Posto Santa Felicia') for i in range(year_start, year_end + 1)],
            ignore_index=True
        )

        return esc_df, posto_df

    elif source == DataSource.MAPLU_USP:
        print("Processando dados do DataSource.MAPLU_USP...")

        # Lê e processa os dados da USP
        usp_df = pd.read_csv(f'{data_path}/MAPLU/USP2.csv')
        usp_df[['Hour', 'Min']] = usp_df.Time.str.split(":", expand=True)
        usp_df = usp_df[['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation']]
        return convert_to_numeric(usp_df, ['Year', 'Month', 'Day', 'Hour', 'Min', 'Precipitation'])
    
    else:
        raise ValueError(f"Fonte '{source}' não suportada.")
