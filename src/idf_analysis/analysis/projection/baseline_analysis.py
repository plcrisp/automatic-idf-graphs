"""
Análise de Dados Climáticos Corretos por Viés - Baseline

Este script realiza uma análise climática baseada em modelos climáticos globais (GCMs) simulando o período de referência (baseline).
Ele aplica diferentes métodos de correção de viés aos dados simulados e gera estatísticas importantes para estudos de mudança climática.

📌 Objetivos principais:
- Processar dados simulados de precipitação de modelos climáticos (baseline)
- Corrigir os dados com diferentes métodos de correção de viés
- Calcular estatísticas como:
    - Percentil 90 (P90) diário
    - Máxima precipitação diária anual
    - Tendência temporal da precipitação

🌍 Modelos Climáticos Utilizados:
- **HADGEM**: Desenvolvido pelo Met Office Hadley Centre (Reino Unido)
- **MIROC5**: Desenvolvido pela Universidade de Tóquio (Japão)
Esses modelos fazem parte de experimentos climáticos globais, como o CMIP, e simulam o clima da Terra com base em equações físicas.

🕰️ Baseline:
O baseline é o período de referência histórico (ex: 1980–2005) simulado pelos modelos climáticos. Ele é usado para validar os modelos contra dados observados e como base de comparação para avaliar mudanças futuras no clima.

🛠️ Métodos de Correção de Viés Aplicados:
- **MD** (*Mean Distribution*): Corrige a média e a distribuição dos dados.
- **PT** (*Power Transformation*): Aplica transformações matemáticas para reduzir o viés.
- **QM** (*Quantile Mapping*): Corrige os quantis da distribuição, muito usado.
- **DBC** (*Double Bias Correction*): Dupla correção que ajusta média e variabilidade.

⚙️ Operações realizadas:
1. Leitura dos arquivos simulados corrigidos por viés (um por modelo e método).
2. Cálculo do P90 para cada série temporal corrigida.
3. Agregação por ano e exportação dos dados anuais.
4. Cálculo da maior precipitação diária em cada ano.
5. Análise de tendência da precipitação total anual e da precipitação máxima diária anual, com base em regressão.

📁 Estrutura de diretórios esperada:
Os arquivos CSV devem estar organizados em:  
`GCM_data/bias_correction/{modelo}_baseline_{método}_daily.csv`

Exemplo de nome de arquivo: `HADGEM_baseline_QM_daily.csv`
"""

from ..historical.extremes import calculate_p90,max_annual_precipitation
from ...data.processing import read_csv, aggregate_to_csv, verification, fill_missing_data
from ..historical.trend import get_trend

import pandas as pd
import datetime
import numpy as np
from typing import Tuple, List, Union



"""
--------------------------------------------------------------------------------------------------------------
-------------------------------------------- ANALISANDO DADOS SIMULADOS --------------------------------------
--------------------------------------------------------------------------------------------------------------
"""


def analyze_baseline_bias_corrected_gcms(
    models: list[str],
    bias_methods: list[str],
    base_path: str = 'GCM_data/bias_correction',
    frequency: str = 'daily',
    group_name: str = 'GCM_baseline',
    alpha: float = 0.05,
    save_csv: bool = True,
):
    """
    Realiza análise de dados simulados no período de baseline com diferentes modelos climáticos e métodos de correção de viés.

    Esta função processa séries temporais diárias simuladas por modelos climáticos globais (GCMs) corrigidos por diversos métodos,
    referentes ao período histórico (baseline). Para cada combinação de modelo e método, calcula-se:

    - Percentil 90 diário (P90)
    - Agregação anual da série
    - Máxima precipitação diária por ano
    - Análise de tendência da precipitação anual e da precipitação máxima diária anual

    Os dados processados podem ser exportados em formato `.csv`, e os resultados estatísticos são exibidos no console.

    Parâmetros:
    ----------
    models : list of str
        Lista dos nomes dos modelos climáticos utilizados (ex: ['HADGEM', 'MIROC5']).

    bias_methods : list of str
        Lista dos métodos de correção de viés aplicados aos dados simulados (ex: ['MD', 'PT', 'QM', 'DBC']).

    base_path : str, default='GCM_data/bias_correction'
        Caminho para o diretório onde estão armazenados os arquivos `.csv` com os dados corrigidos.
        
    frequency : str, default='daily'
        Frequência dos dados a serem analisados.

    group_name : str, default='GCM_baseline'
        Nome do grupo a ser utilizado nas análises de tendência, útil para agrupamentos ou legendas.

    alpha : float, default=0.05
        Nível de significância para o teste estatístico de tendência (ex: 0.05 para 95% de confiança).

    save_csv : bool, default=True
        Se True, os dados agregados por ano e os máximos anuais são salvos como arquivos `.csv`.

    Retorno:
    -------
    None
        Os resultados são impressos no console e, se desejado, arquivos são salvos no diretório especificado.
    """
    
    print('--- Baseline Analysis ---')
    sites_list = []

    for model in models:
        for method in bias_methods:
            name = f"{model}_baseline_{method}"
            file_path = f"{base_path}/{name}_{frequency}.csv"

            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"[!] Arquivo não encontrado: {file_path}")
                continue

            print(f'--> P90 {name}: {calculate_p90(df=df)}')

            if save_csv:
                aggregate_to_csv(df=df,name=name,directory=base_path)

                max_annual_precipitation(df=df,name_file=name,directory=base_path)

            sites_list.append(name)

    print('\n--> Trend analysis')
    print('- Annual precipitation')
    get_trend(var='Year', sites_list=sites_list, group=group_name, alpha=alpha,data_type='mod')

    print('\n- Max_daily')
    get_trend(var='Max_daily', sites_list=sites_list, group=group_name, alpha=alpha, data_type='mod')

    print('\nDone!')
    


def prepare_data_pair(
    path_observed: str,
    path_gcm: str,
    return_dataframes: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Prepara e sincroniza dois conjuntos de dados de precipitação (observado e simulado) para análise conjunta.

    Este processo inclui:
    1. Leitura e verificação dos arquivos CSV com dados diários de precipitação.
    2. Preenchimento de falhas nos dados (gap filling) com base na função `fill_missing_data`.
    3. Conversão das datas para o formato datetime e sincronização de ambos os datasets para um período comum.
    4. Extração dos dados de precipitação e dos rótulos temporais ou retorno dos DataFrames completos.

    Parâmetros:
        path_observed (str): Caminho completo para o arquivo CSV com os dados observados.
        path_gcm (str): Caminho completo para o arquivo CSV com os dados simulados (GCM).
        return_dataframes (bool, opcional): Se True, a função retorna os dois DataFrames sincronizados.
                                             Se False (padrão), retorna os arrays de dados e os rótulos de data.

    Retorna:
        Union[Tuple[np.ndarray, np.ndarray, List[str]], Tuple[pd.DataFrame, pd.DataFrame]]:
            - Se `return_dataframes` for False (padrão):
                - Array com dados de precipitação observada.
                - Array com dados de precipitação simulada (GCM).
                - Lista de datas no formato 'DD-MM-AA' correspondentes às observações.
            - Se `return_dataframes` for True:
                - DataFrame com os dados observados, sincronizado e limpo.
                - DataFrame com os dados do GCM, sincronizado e limpo.
    """
    # --- Leitura, verificação e preenchimento de falhas ---
    df_obs = read_csv(path_observed)
    df_gcm = read_csv(path_gcm)

    verification(df_obs)
    verification(df_gcm)

    df_obs = fill_missing_data(path_main=path_observed)
    df_gcm = fill_missing_data(path_main=path_gcm)

    # --- Conversão, limpeza e criação da coluna de data ---
    for df in [df_obs, df_gcm]:
        df['Precipitation'] = pd.to_numeric(df['Precipitation'], errors='coerce')
        df.dropna(subset=['Precipitation'], inplace=True)
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        
    # --- Sincronização: Determina e filtra pelo intervalo de datas em comum ---
    start_date = max(df_obs['Date'].min(), df_gcm['Date'].min())
    end_date = min(df_obs['Date'].max(), df_gcm['Date'].max())
    
    print(f"Período comum considerado: {start_date.strftime('%d-%m-%Y')} até {end_date.strftime('%d-%m-%Y')}")

    df_obs = df_obs[(df_obs['Date'] >= start_date) & (df_obs['Date'] <= end_date)].copy()
    df_gcm = df_gcm[(df_gcm['Date'] >= start_date) & (df_gcm['Date'] <= end_date)].copy()

    # --- Verificação final e retorno condicional ---
    if not df_obs['Date'].reset_index(drop=True).equals(df_gcm['Date'].reset_index(drop=True)):
        # Nota: reset_index é usado para comparar apenas os valores das datas, ignorando o índice do DataFrame
        raise ValueError("Datas dos datasets não estão alinhadas mesmo após o corte.")

    if return_dataframes:
        # Opção 1: Retorna os DataFrames completos e sincronizados
        return df_obs, df_gcm
    else:
        # Opção 2 (Padrão): Retorna os arrays e os rótulos
        labels = df_obs['Date'].dt.strftime('%d-%m-%y').tolist()
        data_obs = df_obs['Precipitation'].values
        data_gcm = df_gcm['Precipitation'].values
        return data_obs, data_gcm, labels



def load_and_clean_precipitation_data(file_path: str):
    """
    Loads a tab-separated file with columns: year, month, day, precipitation.
    Filters out invalid dates and returns a DataFrame with columns:
    year, month, day, date (datetime), and precipitation.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        pd.DataFrame: Cleaned and sorted DataFrame.
    """

    # Load data
    df = read_csv(file_path)

    # Convert to valid dates, invalid dates become NaT
    def try_parse_date(row):
        try:
            return datetime.date(int(row["Year"]), int(row["Month"]), int(row["Day"]))
        except ValueError:
            return pd.NaT

    df["Date"] = df.apply(try_parse_date, axis=1)

    # Remove invalid dates
    df = df[df["Date"].notna()]

    # Convert date column to datetime64 type for consistency
    df["Date"] = pd.to_datetime(df["Date"])

    # Reset index and reorder columns
    df = df.reset_index(drop=True)
    df = df[["Date", "Precipitation", "Year", "Month", "Day"]]

    return df



def prepare_future_data(path_gcm_future: str, return_dataframes: bool = False):
    df_future = read_csv(path_gcm_future)
    
    df_future = load_and_clean_precipitation_data(path_gcm_future)
    
    # Corrigir separadores incorretos (ponto para milhar, vírgula para decimal)
    df_future['Precipitation'] = (
        df_future['Precipitation']
        .astype(str)
        .str.replace('.', '', regex=False)  # remove milhar
        .str.replace(',', '.', regex=False)  # converte decimal
    )

    # Converte para número
    df_future['Precipitation'] = pd.to_numeric(df_future['Precipitation'], errors='coerce')
    
    verification(df_future)
    
    # Gap filling
    df_future = fill_missing_data(path_main=path_gcm_future)
    
    # Remove valores inválidos
    df_future.dropna(subset=['Precipitation'], inplace=True)
    
    # Cria coluna de data
    df_future['Date'] = pd.to_datetime(df_future[['Year', 'Month', 'Day']])
        
    if return_dataframes:
        # Opção 1: Retorna os DataFrames completos e sincronizados
        return df_future
    else:
        # Opção 2 (Padrão): Retorna os arrays e os rótulos
        labels = df_future['Date'].dt.strftime('%d-%m-%y').tolist()
        data_future = df_future['Precipitation'].values
        return data_future, labels
    
    