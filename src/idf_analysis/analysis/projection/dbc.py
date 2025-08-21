from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .quantile_mapping import prepare_data_pair, prepare_future_data

def dbc_percentilico(
    name_obs: str,
    name_baseline: str,
    name_future: str,
    dir: str = 'results',
    percentis: np.ndarray = np.linspace(0.01, 0.99, 99),
    plot: bool = True,
    save_csv_path: str | None = None
):
    """
    Aplica a correção de viés DBC percentílico à série futura, com base nas distribuições observada e GCM histórica.

    Parâmetros:
        name_obs (str): Nome do arquivo observado (sem sufixo '_daily.csv').
        name_baseline (str): Nome do GCM baseline.
        name_future (str): Nome do GCM futuro.
        dir (str): Diretório do observado.
        percentis (np.ndarray): Vetor de percentis a serem usados.
        plot (bool): Se True, plota o gráfico da correção.
        save_csv_path (str | None): Caminho para salvar o CSV corrigido (opcional).

    Retorna:
        pd.DataFrame com 'Date', 'Precipitation Original', 'Precipitation'.
    """

    # Caminhos
    path_obs = f"{dir}/{name_obs}_daily.csv"
    path_baseline = f"{dir}/{name_baseline}_daily.csv"
    path_future = f"{dir}/{name_future}_daily.csv"

    # Carrega os dados históricos tratados
    data_obs, data_gcm_hist, _ = prepare_data_pair(path_obs, path_baseline)

    # Calcula os valores para cada percentil
    q_obs = np.percentile(data_obs, percentis * 100)
    q_hist = np.percentile(data_gcm_hist, percentis * 100)

    # Diferença percentílica
    delta = q_obs - q_hist

    # Carrega os dados futuros tratados
    data_gcm_future, labels_future = prepare_future_data(path_future)

    # Calcula os percentis de cada valor da série futura em relação ao baseline
    # (usando ECDF empírica invertida)
    ecdf_hist = ECDF(data_gcm_hist)
    percentis_futuros = np.clip(ecdf_hist(data_gcm_future), percentis.min(), percentis.max())

    # Interpola o delta para os percentis futuros
    delta_interpolado = np.interp(percentis_futuros, percentis, delta)

    # Aplica a correção
    data_corrigida = data_gcm_future + delta_interpolado
    data_corrigida[data_corrigida < 0] = 0

    # Monta DataFrame de saída
    df_corrigido = pd.DataFrame({
        'Date': labels_future,
        'Precipitation Original': data_gcm_future,
        'Precipitation': data_corrigida
    })

    # Salva se necessário
    if save_csv_path is not None:
        df_corrigido.to_csv(save_csv_path, index=False)

    # Plot (opcional)
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(percentis * 100, q_obs, label='Observado', linewidth=2)
        plt.plot(percentis * 100, q_hist, label='GCM histórico', linewidth=2)
        plt.plot(percentis * 100, delta, label='Delta (Obs - GCM)', linestyle='--')
        plt.xlabel('Percentil')
        plt.ylabel('Precipitação (mm)')
        plt.title('Correção Percentílica (DBC)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return df_corrigido