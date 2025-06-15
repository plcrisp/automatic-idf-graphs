
"""
Este módulo implementa um método de **correção de viés climático** usando *Quantile Mapping* (Mapeamento de Quantis) 
combinado com uma **regressão log-linear**, visando melhorar a adequação dos dados simulados por
modelos climáticos regionais (RCMs ou GCMs) aos dados observacionais.

## O que é correção de viés?

Modelos climáticos, apesar de úteis para simular cenários futuros, muitas vezes apresentam 
**viés sistemático**, ou seja, tendências consistentes de superestimar ou subestimar variáveis como 
precipitação e temperatura. A **correção de viés (bias correction)** é um conjunto de técnicas 
estatísticas utilizadas para ajustar esses dados simulados com base em registros históricos observados.

## O que é Quantile Mapping?

O *Quantile Mapping* é uma técnica amplamente usada na correção de viés que se baseia na comparação 
entre as distribuições de probabilidade dos dados simulados e os dados observados. Ele consiste em:

1. **Ajustar distribuições teóricas** (ex: Lognormal, Gumbel, GEV) aos dados históricos observados e simulados.
2. Para cada valor simulado, encontrar o seu **quantil correspondente** na distribuição acumulada.
3. Mapear esse quantil para o valor equivalente na distribuição observacional, corrigindo assim o dado.

Esse processo permite que os dados simulados reflitam melhor a magnitude e a frequência das variáveis climáticas reais.

## Método adotado neste script

Este módulo vai além do *Quantile Mapping* tradicional ao incluir:

- Um **ajuste espacial** via mapeamento de quantis entre os dados históricos do modelo e os observacionais.
- Uma **regressão log-linear** entre os dados simulados e os corrigidos, permitindo aplicar a mesma relação aos cenários futuros.
- A possibilidade de visualizar graficamente a qualidade do ajuste com o coeficiente de determinação (R²).

Essa abordagem é especialmente útil para preparar dados climáticos para modelagens hidrológicas, 
estudos de impacto climático e análise de eventos extremos.

"""

from ...data.processing import read_csv, verification, fill_missing_data
from src.idf_analysis.core.distributions import get_top_fitted_distributions,fit_data,get_common_distributions

import pandas as pd
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

"""
--------------------------------------------------------------------------------------------------------------
-------------------------------- QUANTILE MAPPING PARA CORREÇÃO DE VIÉS --------------------------------------
--------------------------------------------------------------------------------------------------------------
"""




def prepare_data_pair(path_observed: str, path_gcm: str):
    """
    Prepara e sincroniza dois conjuntos de dados de precipitação (observado e simulado) para análise conjunta.

    Este processo inclui:
    1. Leitura e verificação dos arquivos CSV com dados diários de precipitação.
    2. Preenchimento de falhas nos dados (gap filling) com base na função `fill_missing_data`.
    3. Conversão das datas para o formato datetime e sincronização de ambos os datasets para um período comum.
    4. Extração dos dados de precipitação e dos rótulos temporais no formato 'DD-MM-AA'.

    Parâmetros:
        path_observed (str): Caminho completo para o arquivo CSV com os dados observados.
        path_gcm (str): Caminho completo para o arquivo CSV com os dados simulados (GCM).

    Retorna:
        Tuple[np.ndarray, np.ndarray, List[str]]:
            - Array com dados de precipitação observada.
            - Array com dados de precipitação simulada (GCM).
            - Lista de datas no formato 'DD-MM-AA' correspondentes às observações sincronizadas.
    """
    # Leitura e verificação
    
    df_obs = read_csv(path_observed)
    df_gcm = read_csv(path_gcm)

    verification(df_obs)
    verification(df_gcm)

    # Gap filling (assume que retorna DataFrame com coluna 'Precipitation')
    df_obs = fill_missing_data(path_main=path_observed)
    df_gcm = fill_missing_data(path_main=path_gcm)

    # Conversão e limpeza
    for df in [df_obs, df_gcm]:
        df['Precipitation'] = pd.to_numeric(df['Precipitation'], errors='coerce')
        df.dropna(subset=['Precipitation'], inplace=True)
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        
    # Determina o intervalo em comum
    start_date = max(df_obs['Date'].min(), df_gcm['Date'].min())
    end_date = min(df_obs['Date'].max(), df_gcm['Date'].max())
    
    print(f"Período comum considerado: {start_date.strftime('%d-%m-%Y')} até {end_date.strftime('%d-%m-%Y')}")

    # Filtra ambos para o mesmo intervalo
    df_obs = df_obs[(df_obs['Date'] >= start_date) & (df_obs['Date'] <= end_date)].copy()
    df_gcm = df_gcm[(df_gcm['Date'] >= start_date) & (df_gcm['Date'] <= end_date)].copy()

    # Verifica se ainda estão sincronizados
    if not df_obs['Date'].equals(df_gcm['Date']):
        raise ValueError("Datas dos datasets não estão alinhadas mesmo após o corte.")

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




def prepare_future_data(path_gcm_future: str):
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
    
    labels = df_future['Date'].dt.strftime('%d-%m-%y').tolist()
    data_future = df_future['Precipitation'].values
    
    
    print(df_future['Precipitation'].describe())
    
    return data_future, labels
    



def get_adjusted_distributions(path_observed: str, path_gcm: str):
    """
    Ajusta distribuições estatísticas teóricas aos dados observados e simulados (GCM) para posterior correção de viés.

    Este processo inclui:
    1. Leitura e sincronização dos dados observados e simulados usando `prepare_data_pair`.
    2. Ajuste de distribuições de probabilidade às séries observadas e simuladas, selecionando a melhor com base em critérios estatísticos.
    3. Criação dos objetos de distribuição ajustados com base nas melhores distribuições encontradas.

    Parâmetros:
        path_observed (str): Caminho completo para o arquivo CSV com os dados observados.
        path_gcm (str): Caminho completo para o arquivo CSV com os dados simulados (GCM baseline).

    Retorna:
        Tuple[scipy.stats.rv_continuous, scipy.stats.rv_continuous, np.ndarray, np.ndarray, List[str]]:
            - Objeto de distribuição ajustada para os dados observados.
            - Objeto de distribuição ajustada para os dados GCM baseline.
            - Série de precipitação observada (numpy array).
            - Série de precipitação simulada (GCM baseline, numpy array).
            - Lista de datas no formato 'DD-MM-AA' correspondente ao período comum.
    """
    
    # Prepara os dados
    data_obs, data_gcm_baseline, labels = prepare_data_pair(path_observed, path_gcm)
    
    # Ajuste das distribuições observadas
    results_obs = fit_data(data_obs)
    dist_obs_df = get_top_fitted_distributions(data_obs, results_obs, n=1)

    # Ajuste das distribuições simuladas (GCM)
    results_gcm = fit_data(data_gcm_baseline)
    dist_gcm_df = get_top_fitted_distributions(data_gcm_baseline, results_gcm, n=1)

    # Lista de distribuições válidas
    common_dists = get_common_distributions()
    DIST_MAP = {dist.name.lower(): dist for dist in common_dists}

    # --- Ajuste GCM ---
    c_gcm = dist_gcm_df['c'].iloc[0]
    loc_gcm = dist_gcm_df['loc'].iloc[0]
    scale_gcm = dist_gcm_df['scale'].iloc[0]
    dist_name_gcm = dist_gcm_df['distribution'].iloc[0].strip().lower()
    dist_func_gcm = DIST_MAP[dist_name_gcm]
    dist_gcm = dist_func_gcm(loc=loc_gcm, scale=scale_gcm) if math.isnan(c_gcm) else dist_func_gcm(c_gcm, loc=loc_gcm, scale=scale_gcm)

    # --- Ajuste Observado ---
    c_obs = dist_obs_df['c'].iloc[0]
    loc_obs = dist_obs_df['loc'].iloc[0]
    scale_obs = dist_obs_df['scale'].iloc[0]
    dist_name_obs = dist_obs_df['distribution'].iloc[0].strip().lower()
    dist_func_obs = DIST_MAP[dist_name_obs]
    dist_obs = dist_func_obs(loc=loc_obs, scale=scale_obs) if math.isnan(c_obs) else dist_func_obs(c_obs, loc=loc_obs, scale=scale_obs)

    # Retorna os dois objetos de distribuição ajustados
    return dist_obs, dist_gcm, data_obs, data_gcm_baseline, labels



def quantile_mapping(name_obs: str, name_gcm_baseline: str, name_gcm_future: str,  dir_obs: str = 'results', dir_gcm: str = '../datasets/GCM', plot=True, save_csv_path: str = None):
    """
    Executa a correção de viés de modelos climáticos via Quantile Mapping com ajuste espacial e regressão log-linear.

    Este processo envolve:
    1. Ajuste de distribuições estatísticas teóricas (ex: GEV, Lognormal) aos dados observados e simulados históricos.
    2. Mapeamento de quantis (spatial downscaling) entre a distribuição do GCM e a distribuição observada.
    3. Ajuste de regressão log-linear entre os dados GCM históricos e os espacialmente corrigidos.
    4. Aplicação dessa regressão aos dados GCM futuros para gerar a série corrigida.
    5. Geração opcional de gráfico com a regressão e o coeficiente de determinação (R²).

    Parâmetros:
        name_obs (str): Nome base do arquivo CSV com os dados observados (sem sufixo '_daily.csv').
        name_gcm_baseline (str): Nome base do arquivo CSV com os dados GCM históricos.
        name_gcm_future (str): Nome base do arquivo CSV com os dados GCM projetados para o futuro.
        dir_obs (str): Diretório onde está localizado o arquivo observado. Padrão: 'results'.
        dir_gcm (str): Diretório onde estão os arquivos GCM. Padrão: 'GCM_data/bias_correction'.
        plot (bool): Se True, plota a regressão log-linear. Padrão: True.
        save_csv_path (str | None): Caminho para salvar o CSV com os dados corrigidos. Se None, não salva. Padrão: None.

    Retorna:
        pandas.DataFrame:
            DataFrame contendo:
            - 'Date': Datas correspondentes à série futura.
            - 'Precipitation Original': Precipitação original simulada do GCM futuro.
            - 'Precipitation': Precipitação futura corrigida (bias-corrected).
    """
    
    path_observed = f'{dir_obs}/{name_obs}_daily.csv'
    path_gcm_baseline = f'{dir_gcm}/{name_gcm_baseline}_daily.csv'
    path_gcm_future = f'{dir_gcm}/{name_gcm_future}_daily.csv'
    
    
    # Etapa 1: Ajuste das distribuições
    dist_obs, dist_gcm, data_obs, data_gcm_baseline, labels = get_adjusted_distributions(path_observed, path_gcm_baseline)

    # Etapa 2: Correção espacial por mapeamento de quantis (baseline GCM -> observacional)
    inv_cdf_gcm = dist_gcm.cdf(data_gcm_baseline)
    data_spatdown = dist_obs.ppf(inv_cdf_gcm)
    data_spatdown[data_spatdown < 0] = 0  # remove negativos

    # Etapa 3: Regressão log-linear entre GCM baseline e série corrigida
    x = np.log1p(data_gcm_baseline)
    y = data_spatdown

    coefs = np.polyfit(x, y, 1)
    a, b = coefs
    model = np.poly1d(coefs)
    r2 = r2_score(y, model(x))
    
    data_gcm_future, labels_future = prepare_future_data(path_gcm_future)

    # Etapa 4: Aplicação ao cenário futuro
    x_future = np.log1p(data_gcm_future)
    data_corrected = model(x_future)
    data_corrected[data_corrected < 0] = 0

    # Plot opcional
    if plot:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, alpha=0.5, label='GCM baseline vs. SpatDown')
        plt.plot(x, model(x), color='red', label=f'Regressão: y = {a:.2f}x + {b:.2f}\n$R^2$ = {r2:.3f}')
        plt.xlabel('log(Precipitação GCM baseline)')
        plt.ylabel('Precipitação corrigida (5min)')
        plt.title('Regressão para Correção Espacial')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    df_corrected = pd.DataFrame({
        'Date': labels_future,
        'Precipitation Original': data_gcm_future,
        'Precipitation': data_corrected
    })
    
    # Salva CSV se caminho for informado
    if save_csv_path is not None:
        df_corrected.to_csv(save_csv_path, index=False)

    return df_corrected
    
    


quantile_mapping(name_obs='inmet_santana', name_gcm_baseline='HADGEM_baseline', name_gcm_future='HADGEM_rcp45', dir_obs='results/inmet_santana', dir_gcm='datasets/GCM', plot=True, save_csv_path='results/inmet_santana/inmet_santana_future.csv')
