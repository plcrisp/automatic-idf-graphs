import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple
from sklearn.metrics import r2_score
from pathlib import Path

from .baseline_analysis import prepare_data_pair, prepare_future_data
from ...analysis.historical.validation import max_annual_precipitation
from ...analysis.historical.intervals import get_subdaily_from_disaggregation_factors, DisaggregationScenario
from ...core.distributions import get_distribution

# --- Constante do Módulo ---
COLUNAS_DADOS = [
    'Max_5min', 'Max_10min', 'Max_15min', 'Max_20min', 'Max_25min', 
    'Max_30min', 'Max_1h', 'Max_6h', 'Max_8h', 'Max_10h', 'Max_12h', 'Max_24h'
]



def load_and_prepare_data(name_obs, name_gcm_baseline, name_gcm_future, dir):
    """Carrega, prepara e extrai os máximos anuais dos dados observados e do GCM."""
    print("\nPasso 1: Carregando e preparando os dados anuais...")
    
    # Caminhos para os arquivos de dados
    path_obs = f"{dir}/{name_obs}_daily.csv"
    path_gcm_baseline = f"{dir}/{name_gcm_baseline}_daily.csv"
    path_gcm_future = f"{dir}/{name_gcm_future}_daily.csv"
    
    print('\n[INFO] Preparando dados históricos.')
    
    # Prepara pares de dados observados e de baseline
    df_obs, df_baseline = prepare_data_pair(path_observed=path_obs, path_gcm=path_gcm_baseline, return_dataframes=True)
    
    print('\n[INFO] Preparando dados futuros.')
    
    df_future = prepare_future_data(path_gcm_future=path_gcm_future, return_dataframes=True)
    
    # Calcula a precipitação máxima anual para cada conjunto de dados
    obs_max = max_annual_precipitation(df_obs, name_file=name_obs, output_dir=dir)
    baseline_max = max_annual_precipitation(df_baseline, name_file=name_gcm_baseline, output_dir=dir)
    future_max = max_annual_precipitation(df_future, name_file=name_gcm_future, output_dir=dir)
    
    return obs_max, baseline_max, future_max, df_obs, df_baseline, df_future



def fit_distributions(df_obs, obs_max, baseline_max, future_max):
    """Ajusta as distribuições de probabilidade para todos os conjuntos de dados."""
    print("\nPasso 2: Ajustando as distribuições de probabilidade...")

    # Ajusta distribuições para os dados de máximos anuais do GCM
    dist_baseline = get_distribution(data_df=baseline_max, column_name='Precipitation', n=1, plot=False)
    dist_future = get_distribution(data_df=future_max, column_name='Precipitation', n=1, plot=False)
    
    # Ajusta distribuição para dados observados (máximos anuais)
    dist_obs_annual = get_distribution(data_df=obs_max, column_name='Precipitation', n=1, plot=False)

    # Ajusta uma distribuição para cada duração sub-diária dos dados observados
    dists_hist = {}
    for duracao in COLUNAS_DADOS:
        dist_obj = get_distribution(
            data_df=df_obs,
            column_name=duracao, 
            n=1, 
            plot=False
        )
        if dist_obj is None:
            raise ValueError(f"[ERRO] Não foi possível reconstruir a distribuição para {duracao}")
        dists_hist[duracao] = dist_obj
        
    return dist_baseline, dist_future, dists_hist, dist_obs_annual



def perform_spatial_downscaling(dist_baseline, dists_hist, data_baseline):
    """Executa o downscaling espacial para o período de baseline."""
    print("\nPasso 3: Executando Downscaling Espacial (Baseline)...")
    
    prob_baseline = dist_baseline.cdf(data_baseline)
    
    dados_spatdown = {}
    for duracao in COLUNAS_DADOS:
        # Mapeia as probabilidades do GCM para os quantis da distribuição observada
        dados_spatdown[duracao] = dists_hist[duracao].ppf(prob_baseline)
        
    return dados_spatdown



def calculate_regression_coefficients(x_data, y_spatial_data, y_temporal_data, verbose=True):
    """Calcula os coeficientes de regressão linear para os downscalings."""
    print("\nPasso 4: Calculando os coeficientes de regressão...")
    
    # Regressao linear simples (sem transformacao logaritmica)
    x = x_data
    
    # --- Coeficientes para o Downscaling Espacial (a1, b1) ---
    coefs_spat = {}
    if verbose:
        print("\n  Coeficientes do Downscaling Espacial (a1, b1):")
    for duracao in COLUNAS_DADOS:
        y = y_spatial_data[duracao]
        # polyfit de grau 1 é equivalente a uma regressão linear simples
        a1, b1 = np.polyfit(x, y, 1)
        r_quadrado = r2_score(y, np.poly1d([a1, b1])(x))
        coefs_spat[duracao] = {'a1': a1, 'b1': b1}
        if verbose:
            print(f"    Duração {duracao:<10} -> a1: {a1:<7.4f}, b1: {b1:<7.4f}, R²: {r_quadrado:.4f}")

    # --- Coeficientes para o Downscaling Temporal (a2, b2) ---
    a2, b2 = np.polyfit(x, y_temporal_data, 1)
    r_quadrado_temp = r2_score(y_temporal_data, np.poly1d([a2, b2])(x))
    coefs_temp = {'a2': a2, 'b2': b2}
    if verbose:
        print("\n  Coeficientes do Downscaling Temporal (a2, b2):")
        print(f"    a2: {a2:<7.4f}, b2: {b2:<7.4f}, R²: {r_quadrado_temp:.4f}")
        
    return coefs_spat, coefs_temp



def apply_eqm_correction(data_series, temporal_coeffs, spatial_coeffs):
    """Aplica a correcao EQM para gerar valores sub-diarios a partir de uma serie diaria."""
    print("\nPasso 5: Aplicando correcao EQM...")
    
    a2, b2 = temporal_coeffs['a2'], temporal_coeffs['b2']
    
    # Etapa chave do EQM:
    # 1. Inverte a relacao temporal para encontrar o "equivalente" de um valor futuro no periodo de baseline.
    #    Formula original: Y_temp = a2*X_base + b2  =>  X_base = (Y_temp - b2) / a2
    gcm_equivalente_baseline = (data_series - b2) / a2
    
    # 2. Aplica a relacao espacial a este valor "equivalente" para obter o resultado final.
    #    Formula: Y_final = a1*X_base + b1
    dados_finais_futuro = {}
    for duracao in COLUNAS_DADOS:
        a1 = spatial_coeffs[duracao]['a1']
        b1 = spatial_coeffs[duracao]['b1']
        dados_finais_futuro[duracao] = a1 * gcm_equivalente_baseline + b1
        
    return dados_finais_futuro



def save_results(
    anos_baseline, 
    data_baseline, 
    dados_spatdown, 
    anos_future, 
    data_future, 
    dados_finais_futuro, 
    name_gcm_baseline, 
    name_gcm_future, 
    name_obs, 
    disag_factor_str, 
    dir_gcm
):
    """Salva os resultados dos períodos de baseline e futuro em arquivos CSV."""
    print("\nPasso 6: Salvando resultados em arquivos CSV...")

    dir_gcm = Path(dir_gcm)

    # --- DataFrame para o período BASELINE ---
    dict_baseline = {'Year': anos_baseline, 'baseline_daily': data_baseline, **dados_spatdown}
    df_baseline = pd.DataFrame(dict_baseline).round(2)

    path_baseline = dir_gcm / f"max_subdaily_{name_gcm_baseline}_{name_obs}_{disag_factor_str}_baseline.csv"
    path_baseline.parent.mkdir(parents=True, exist_ok=True) 
    df_baseline.to_csv(path_baseline, index=False)
    print(f"  -> Salvo: {path_baseline}")

    # --- DataFrame para o período FUTURO ---
    dict_futuro = {'Year': anos_future, 'future_daily': data_future, **dados_finais_futuro}
    df_futuro = pd.DataFrame(dict_futuro).round(2)

    path_futuro = dir_gcm / f"max_subdaily_{name_gcm_future}_{name_obs}_{disag_factor_str}_future.csv"
    path_futuro.parent.mkdir(parents=True, exist_ok=True) 
    df_futuro.to_csv(path_futuro, index=False)
    print(f"  -> Salvo: {path_futuro}")
    


def generate_eqm_figure_side_by_side(
    obs_max: pd.DataFrame,
    baseline_max: pd.DataFrame,
    corrected_baseline: np.ndarray
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Gera uma única figura com dois subplots relacionados ao processo de downscaling via EQM:
    
    1. Comparação CDF→CDF entre dados observados e dados de baseline do GCM.
    2. Série temporal: Baseline original, Observado e Baseline corrigido.
    
    Args:
        obs_max (pd.DataFrame): Dados observados históricos de precipitação.
        baseline_max (pd.DataFrame): Dados de precipitação do GCM no período de baseline.
        corrected_baseline (np.ndarray): Dados de baseline do GCM corrigidos pelo EQM.
    
    Returns:
        Tuple[plt.Figure, np.ndarray]: 
            fig - Figura contendo os dois subplots.
            axes - Array de eixos [ax_cdf, ax_series].
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.ecdfplot(obs_max['Precipitation'], label='Observed', ax=axes[0])
    sns.ecdfplot(baseline_max['Precipitation'], label='GCM Baseline', ax=axes[0])
    axes[0].set_xlabel("Precipitation (mm/year)")
    axes[0].set_ylabel("Cumulative Frequency")
    axes[0].set_title("EQM - Observed vs GCM Baseline CDF")
    axes[0].legend()

    years_baseline = baseline_max['Year'].to_numpy()
    years_obs = obs_max['Year'].to_numpy()
    
    axes[1].plot(years_baseline, baseline_max['Precipitation'], label='GCM Baseline (original)', marker='o', alpha=0.7)
    axes[1].plot(years_obs, obs_max['Precipitation'].to_numpy(), label='Observed', marker='^', alpha=0.7)
    axes[1].plot(years_baseline, corrected_baseline, label='GCM Baseline (bias-corrected)', marker='s', alpha=0.7)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Precipitation (mm/year)")
    axes[1].set_title("GCM Baseline - Before vs After Bias Correction")
    axes[1].legend()

    fig.tight_layout()
    return fig, axes



def eqm_downscaling(name_obs: str, name_baseline: str, name_future: str, 
                        scenario: DisaggregationScenario = DisaggregationScenario.BASE, 
                        disag_factor: float = 0.2, dir: str = 'results', 
                        verbose: bool = False, plot: bool = False):
    """
    Executa o downscaling estatístico de dados de chuva de um GCM usando o método
    de Pareamento de Quantis Equidistantes (EQM).

    O objetivo é transformar previsões de chuva máxima diária de um modelo climático
    em previsões de chuva máxima para durações mais curtas (sub-diárias), usando
    como referência dados históricos observados de um local.

    Args:
        name_obs (str): Nome base do arquivo de dados observados.
        name_baseline (str): Nome base do arquivo do GCM para o período de baseline.
        name_future (str): Nome base do arquivo do GCM para o período futuro.
        scenario (DisaggregationScenario): Cenário de desagregação (BASE, UMIDO, SECO).
        disag_factor (float): Fator de desagregação para cenários úmidos/secos.
        dir (str): Diretório dos dados de entrada e saída.
        verbose (bool): Se True, imprime detalhes do processo (ex: coeficientes).
        plot (bool): Se True, gera figura comparando observado vs baseline corrigido no segundo painel.
    """
    print("Iniciando o processo de Downscaling EQM...")
    
    # Define o sufixo do arquivo com base no cenário
    if scenario == DisaggregationScenario.UMIDO:
        disag_factor_str = f'p{disag_factor}'
    elif scenario == DisaggregationScenario.SECO:
        disag_factor_str = f'm{disag_factor}'
    else:
        disag_factor_str = 'ger'
        
    # ETAPA 1: Carregar e Preparar Dados
    obs_max, baseline_max, future_max, df_obs, df_baseline, df_future = load_and_prepare_data(
        name_obs, name_baseline, name_future, dir
    )
    
    # Gera os dados sub-diarios observados (necessario para o ajuste de distribuicao)
    subdaily_obs = get_subdaily_from_disaggregation_factors(df=obs_max, scenario=scenario, var_value=disag_factor, name_file=name_obs, output_dir=dir)

    # ETAPA 2: Ajustar Distribuições de Probabilidade
    dist_baseline, dist_future, dists_hist, dist_obs_annual = fit_distributions(
        subdaily_obs, obs_max, baseline_max, future_max
    )
    
    data_baseline = baseline_max['Precipitation'].to_numpy()
    data_future = future_max['Precipitation'].to_numpy()
    
    # ETAPA 3: Downscaling Espacial e Temporal via Quantis
    dados_spatdown = perform_spatial_downscaling(dist_baseline, dists_hist, data_baseline)
    dados_tempdown = dist_future.ppf(dist_baseline.cdf(data_baseline))
    
    # Criar correcao do baseline para visualizacao (escala de maximos anuais)
    prob_baseline_annual = dist_baseline.cdf(data_baseline)
    baseline_corrected_annual = dist_obs_annual.ppf(prob_baseline_annual)
    
    # ETAPA 4: Calcular Coeficientes de Regressão
    coefs_spat, coefs_temp = calculate_regression_coefficients(
        x_data=data_baseline,
        y_spatial_data=dados_spatdown,
        y_temporal_data=dados_tempdown,
        verbose=verbose
    )
    
    # ETAPA 5: Aplicar correcao EQM nos dados futuros
    dados_finais_futuro = apply_eqm_correction(data_future, coefs_temp, coefs_spat)
    
    # ETAPA 6: Salvar Resultados
    save_results(
        anos_baseline=baseline_max['Year'].to_list(),
        data_baseline=data_baseline,
        dados_spatdown=dados_spatdown,
        anos_future=future_max['Year'].to_list(),
        data_future=data_future,
        dados_finais_futuro=dados_finais_futuro,
        name_gcm_baseline=name_baseline,
        name_gcm_future=name_future,
        name_obs=name_obs,
        disag_factor_str=disag_factor_str,
        dir_gcm=dir
    )
    
    print("\nProcesso de Downscaling EQM concluído com sucesso!")
    
    if plot:
        figs = generate_eqm_figure_side_by_side(obs_max, baseline_max, baseline_corrected_annual)

        return figs