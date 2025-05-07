
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

from ..utils.extreme_precipitation_analysis import calculate_p90,max_annual_precipitation
from ..utils.data_processing import aggregate_to_csv,read_csv
from ..utils.error_correction import verification, fill_missing_data
from ..utils.quality_analysis import get_trend
from ..utils.get_distribution import get_top_fitted_distributions,fit_data,get_common_distributions

import pandas as pd
import math
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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
    df_obs = fill_missing_data(path_main=df_obs)
    df_gcm = fill_missing_data(path_main=df_gcm)

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



def quantile_mapping(name_obs: str, name_gcm_baseline: str, name_gcm_future: str,  dir_obs: str = 'results', dir_gcm: str = 'GCM_data/bias_correction', plot=True):
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
    x = np.log(data_gcm_baseline + 1e-6)  # evita log(0)
    y = data_spatdown

    coefs = np.polyfit(x, y, 1)
    a, b = coefs
    model = np.poly1d(coefs)
    r2 = r2_score(y, model(x))
    
    data_obs, data_gcm_future = prepare_data_pair(path_observed, path_gcm_future)

    # Etapa 4: Aplicação ao cenário futuro
    x_future = np.log(data_gcm_future + 1e-6)
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
        'Date': labels,
        'Precipitation Original': data_gcm_future,
        'Precipitation': data_corrected
    })

    return df_corrected
    
    


    
    
    