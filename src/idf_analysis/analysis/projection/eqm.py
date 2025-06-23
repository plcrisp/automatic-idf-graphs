import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score

from .baseline_analysis import prepare_data_pair, prepare_future_data
from ...analysis.historical.extremes import max_annual_precipitation
from ...analysis.historical.intervals import get_subdaily_from_disaggregation_factors, DisaggregationScenario
from ...core.distributions import get_distribution

# --- Constante do Módulo ---
COLUNAS_DADOS = [
    'Max_5min', 'Max_10min', 'Max_15min', 'Max_20min', 'Max_25min', 
    'Max_30min', 'Max_1h', 'Max_6h', 'Max_8h', 'Max_10h', 'Max_12h', 'Max_24h'
]



def load_and_prepare_data(name_obs, name_gcm_baseline, name_gcm_future, dir_obs, dir_gcm):
    """Carrega, prepara e extrai os máximos anuais dos dados observados e do GCM."""
    print("\nPasso 1: Carregando e preparando os dados anuais...")
    
    # Caminhos para os arquivos de dados
    path_obs = os.path.join(dir_obs, f'{name_obs}_daily.csv')
    path_gcm_baseline = os.path.join(dir_gcm, f'{name_gcm_baseline}_daily.csv')
    path_gcm_future = os.path.join(dir_gcm, f'{name_gcm_future}_daily.csv')
    
    # Prepara pares de dados observados e de baseline
    df_obs, df_baseline = prepare_data_pair(path_observed=path_obs, path_gcm=path_gcm_baseline, return_dataframes=True)
    df_future = prepare_future_data(path_gcm_future=path_gcm_future, return_dataframes=True)
    
    # Calcula a precipitação máxima anual para cada conjunto de dados
    obs_max = max_annual_precipitation(df_obs, name_file=name_obs, output_dir=dir_obs, outliers=True)
    baseline_max = max_annual_precipitation(df_baseline, name_file=name_gcm_baseline, output_dir=dir_gcm, outliers=True)
    future_max = max_annual_precipitation(df_future, name_file=name_gcm_future, output_dir=dir_gcm, outliers=True)
    
    return obs_max, baseline_max, future_max



def fit_distributions(name_obs, name_gcm_baseline, name_gcm_future, disag_factor_str, dir_obs, dir_gcm):
    """Ajusta as distribuições de probabilidade para todos os conjuntos de dados."""
    print("\nPasso 2: Ajustando as distribuições de probabilidade...")

    # Ajusta distribuições para os dados diários do GCM
    dist_baseline = get_distribution(name_file=name_gcm_baseline, n=1, directory=dir_gcm, plot=False)
    dist_future = get_distribution(name_file=name_gcm_future, n=1, directory=dir_gcm, plot=False)

    # Ajusta uma distribuição para cada duração sub-diária dos dados observados
    dists_hist = {}
    for duracao in COLUNAS_DADOS:
        dists_hist[duracao] = get_distribution(
            name_file=name_obs, 
            duration=duracao, 
            disag_factor=disag_factor_str, 
            n=1, 
            directory=dir_obs, 
            plot=False
        )
        
    return dist_baseline, dist_future, dists_hist



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
    
    # A variável 'x' é o logaritmo dos dados do GCM para linearizar a relação
    x = np.log(x_data)
    
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



def project_future_values(data_future, temporal_coeffs, spatial_coeffs):
    """Projeta os dados futuros sub-diários usando os coeficientes de regressão."""
    print("\nPasso 5: Gerando projeções finais para o cenário futuro...")
    
    a2, b2 = temporal_coeffs['a2'], temporal_coeffs['b2']
    
    # Etapa chave do EQM:
    # 1. Inverte a relação temporal para encontrar o "equivalente" de um valor futuro no período de baseline.
    #    Fórmula original: Y_temp = a2*X_base + b2  =>  X_base = (Y_temp - b2) / a2
    gcm_futuro_equivalente_baseline = (data_future - b2) / a2
    
    # 2. Aplica a relação espacial a este valor "equivalente" para obter o resultado final.
    #    Fórmula: Y_final = a1*X_base + b1
    dados_finais_futuro = {}
    for duracao in COLUNAS_DADOS:
        a1 = spatial_coeffs[duracao]['a1']
        b1 = spatial_coeffs[duracao]['b1']
        dados_finais_futuro[duracao] = a1 * gcm_futuro_equivalente_baseline + b1
        
    return dados_finais_futuro



def save_results(anos_baseline, data_baseline, dados_spatdown, anos_future, data_future, dados_finais_futuro, name_gcm_baseline, name_gcm_future, name_obs, disag_factor_str, dir_gcm):
    """Salva os resultados dos períodos de baseline e futuro em arquivos CSV."""
    print("\nPasso 6: Salvando resultados em arquivos CSV...")

    # --- DataFrame para o período BASELINE ---
    dict_baseline = {'Year': anos_baseline, 'baseline_daily': data_baseline, **dados_spatdown}
    df_baseline = pd.DataFrame(dict_baseline).round(2)
    path_baseline = os.path.join(dir_gcm, f'max_subdaily_{name_gcm_baseline}_{name_obs}{disag_factor_str}_baseline.csv')
    df_baseline.to_csv(path_baseline, index=False)
    print(f"  -> Salvo: {path_baseline}")

    # --- DataFrame para o período FUTURO ---
    dict_futuro = {'Year': anos_future, 'future_daily': data_future, **dados_finais_futuro}
    df_futuro = pd.DataFrame(dict_futuro).round(2)
    path_futuro = os.path.join(dir_gcm, f'max_subdaily_{name_gcm_future}_{name_obs}{disag_factor_str}_future.csv')
    df_futuro.to_csv(path_futuro, index=False)
    print(f"  -> Salvo: {path_futuro}")



def eqm_downscaling(name_obs: str, name_gcm_baseline: str, name_gcm_future: str, 
                        scenario: DisaggregationScenario = DisaggregationScenario.BASE, 
                        disag_factor: float = 0.2, dir_obs: str = 'results', 
                        dir_gcm: str = '../datasets/GCM', verbose: bool = True):
    """
    Executa o downscaling estatístico de dados de chuva de um GCM usando o método
    de Pareamento de Quantis Equidistantes (EQM).

    O objetivo é transformar previsões de chuva máxima diária de um modelo climático
    em previsões de chuva máxima para durações mais curtas (sub-diárias), usando
    como referência dados históricos observados de um local.

    Args:
        name_obs (str): Nome base do arquivo de dados observados.
        name_gcm_baseline (str): Nome base do arquivo do GCM para o período de baseline.
        name_gcm_future (str): Nome base do arquivo do GCM para o período futuro.
        scenario (DisaggregationScenario): Cenário de desagregação (BASE, UMIDO, SECO).
        disag_factor (float): Fator de desagregação para cenários úmidos/secos.
        dir_obs (str): Diretório dos dados observados.
        dir_gcm (str): Diretório dos dados do GCM.
        verbose (bool): Se True, imprime detalhes do processo (ex: coeficientes).
    """
    print("Iniciando o processo de Downscaling EQM...")
    
    # Define o sufixo do arquivo com base no cenário
    if scenario == DisaggregationScenario.UMIDO:
        disag_factor_str = f'_p_{disag_factor}'
    elif scenario == DisaggregationScenario.SECO:
        disag_factor_str = f'_m_{disag_factor}'
    else:
        disag_factor_str = '_ger'
        
    # ETAPA 1: Carregar e Preparar Dados
    obs_max, baseline_max, future_max = load_and_prepare_data(
        name_obs, name_gcm_baseline, name_gcm_future, dir_obs, dir_gcm
    )
    
    # Gera os dados sub-diários observados (necessário para o ajuste de distribuição)
    get_subdaily_from_disaggregation_factors(df=obs_max, scenario=scenario, var_value=disag_factor, name_file=name_obs, directory=dir_obs)

    # ETAPA 2: Ajustar Distribuições de Probabilidade
    dist_baseline, dist_future, dists_hist = fit_distributions(
        name_obs, name_gcm_baseline, 
        name_gcm_future, disag_factor_str, dir_obs, dir_gcm
    )
    
    data_baseline = baseline_max['Precipitation'].to_numpy()
    data_future = future_max['Precipitation'].to_numpy()
    
    # ETAPA 3: Downscaling Espacial e Temporal via Quantis
    dados_spatdown = perform_spatial_downscaling(dist_baseline, dists_hist, data_baseline)
    dados_tempdown = dist_future.ppf(dist_baseline.cdf(data_baseline)) # Sinal climático
    
    # ETAPA 4: Calcular Coeficientes de Regressão
    coefs_spat, coefs_temp = calculate_regression_coefficients(
        x_data=data_baseline,
        y_spatial_data=dados_spatdown,
        y_temporal_data=dados_tempdown,
        verbose=verbose
    )
    
    # ETAPA 5: Projetar Valores Futuros
    dados_finais_futuro = project_future_values(data_future, coefs_temp, coefs_spat)
    
    # ETAPA 6: Salvar Resultados
    save_results(
        anos_baseline=baseline_max['Year'].to_list(),
        data_baseline=data_baseline,
        dados_spatdown=dados_spatdown,
        anos_future=future_max['Year'].to_list(),
        data_future=data_future,
        dados_finais_futuro=dados_finais_futuro,
        name_gcm_baseline=name_gcm_baseline,
        name_gcm_future=name_gcm_future,
        name_obs=name_obs,
        disag_factor_str=disag_factor_str,
        dir_gcm=dir_gcm
    )
    
    print("\nProcesso de Downscaling EQM concluído com sucesso!")