
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

from ...core.distributions import get_top_fitted_distributions,fit_data,get_common_distributions
from .baseline_analysis import prepare_data_pair, prepare_future_data

import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

"""
--------------------------------------------------------------------------------------------------------------
-------------------------------- QUANTILE MAPPING PARA CORREÇÃO DE VIÉS --------------------------------------
--------------------------------------------------------------------------------------------------------------
"""




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



def quantile_mapping(name_obs: str, name_baseline: str, name_future: str,  dir: str = 'results', plot=True, save_csv_path: str = None):
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
        name_baseline (str): Nome base do arquivo CSV com os dados GCM históricos.
        name_future (str): Nome base do arquivo CSV com os dados GCM projetados para o futuro.
        dir (str): Diretório onde está localizado o arquivo observado. Padrão: 'results'.
        plot (bool): Se True, plota a regressão log-linear. Padrão: True.
        save_csv_path (str | None): Caminho para salvar o CSV com os dados corrigidos. Se None, não salva. Padrão: None.

    Retorna:
        pandas.DataFrame:
            DataFrame contendo:
            - 'Date': Datas correspondentes à série futura.
            - 'Precipitation Original': Precipitação original simulada do GCM futuro.
            - 'Precipitation': Precipitação futura corrigida (bias-corrected).
    """
    
    path_observed = f'{dir}/{name_obs}_daily.csv'
    path_gcm_baseline = f'{dir}/{name_baseline}_daily.csv'
    path_gcm_future = f'{dir}/{name_future}_daily.csv'
    
    
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
        plt.figure(figsize=(10, 6))
        
        max_val = max(np.max(data_obs), np.max(data_gcm_future), np.max(data_corrected))
        bins = np.linspace(0, min(max_val, np.percentile(np.concatenate([data_obs, data_gcm_future, data_corrected]), 99)), 50)
        
        plt.hist(data_obs, bins=bins, alpha=0.7, label='Observado (referência)', 
                color='green', density=True, edgecolor='darkgreen', linewidth=0.5)
        
        plt.hist(data_gcm_future, bins=bins, alpha=0.6, label='GCM Original (com viés)', 
                color='red', density=True, edgecolor='darkred', linewidth=0.5)
        
        plt.hist(data_corrected, bins=bins, alpha=0.8, label='GCM Corrigido', 
                color='blue', density=True, edgecolor='darkblue', linewidth=0.5)
        
        stats_text = f"""Estatísticas:
                            Observado:     μ={np.mean(data_obs):.2f}, σ={np.std(data_obs):.2f}
                            GCM Original:  μ={np.mean(data_gcm_future):.2f}, σ={np.std(data_gcm_future):.2f}
                            GCM Corrigido: μ={np.mean(data_corrected):.2f}, σ={np.std(data_corrected):.2f}
                            Redução do viés na média: {abs(np.mean(data_gcm_future) - np.mean(data_obs)) - abs(np.mean(data_corrected) - np.mean(data_obs)):.2f}"""
        
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=9, family='monospace')
        
        plt.xlabel('Precipitação (mm)', fontsize=12)
        plt.ylabel('Densidade de Probabilidade', fontsize=12)
        plt.title('Correção de Viés por Quantile Mapping\n' + 
                 'Verde=Meta | Vermelho=Problema | Azul=Solução', fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(0.97, 0.65))
        plt.grid(True, alpha=0.3)
        
        # Melhorar aparência
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
    
    
