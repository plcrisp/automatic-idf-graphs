import os
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Literal, List, Tuple, Optional, Dict
import pandas as pd

from ...data.processing import verification, fill_missing_data, read_csv, remove_outliers_from_max
from ...core.correlation import left_join_precipitation


def p90(df, ax=None, display=True, show_p90=True):
    """
    Plota o gráfico da probabilidade acumulada de não excedência (CDF)
    com base nos dados de precipitação, com a opção de destacar o valor P90.

    Parâmetros:
    - df (pd.DataFrame): DataFrame com uma coluna 'Precipitation'.
    - ax (matplotlib.axes.Axes): Eixo para desenhar o gráfico. Se None, cria um novo.
    - display (bool): Se True, exibe o gráfico com plt.show().
    - show_p90 (bool): Se True, destaca o valor do percentil 90% (P90).

    Retorna:
    - Tuple[float, matplotlib.axes.Axes]: Valor P90 e eixo com o gráfico plotado.
    """

    # Prepara os dados
    df = df[['Precipitation']].query('Precipitation != 0').sort_values('Precipitation').reset_index(drop=True)
    df['Probability'] = (df.index + 1) / len(df) * 100

    # Calcula o valor P90
    p90_value = df.loc[df['Probability'] >= 90, 'Precipitation'].iloc[0]

    # Prepara o eixo
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    # Plota a CDF
    sns.lineplot(x='Probability', y='Precipitation', data=df, ax=ax, color='black')
    ax.set_ylabel('Precipitação (mm)')
    ax.set_xlabel('Probabilidade (%)')
    ax.set_title('Probabilidade de Não-Excedência')

    # Destaca o P90
    if show_p90:
        ax.axhline(p90_value, color='red', linestyle='--', linewidth=1)
        ax.annotate(f'P90 = {p90_value:.2f} mm',
                    xy=(91, p90_value),
                    xytext=(92, p90_value + 2),
                    fontsize=9,
                    color='red',
                    arrowprops=dict(arrowstyle='->', color='red'),
                    bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))

    # Exibe o gráfico se necessário
    if display:
        plt.show()

    return p90_value, ax



def max_annual_precipitation(df, name_file, output_dir='Results', frequency: Literal['daily', 'hourly'] = 'daily', outliers: bool = False):
    """
    Calcula o valor máximo de precipitação anual e remove outliers.
    Para dados horários, soma a precipitação por dia antes de calcular os máximos.

    Parâmetros:
    - df (DataFrame): Deve conter colunas 'Year', 'Precipitation', e dependendo da frequência: 'Month', 'Day', 'Hour'.
    - name_file (str): Nome base do arquivo de saída (sem extensão).
    - output_dir (str): Diretório onde o CSV será salvo.
    - frequency (str): 'daily' (espera valores diários) ou 'hourly' (soma por dia antes de agrupar por ano).
    - outliers (bool): Se True, remove outliers usando a função auxiliar.

    Retorna:
    - DataFrame com os valores máximos de precipitação anual, excluindo outliers.
    """
    print(f"\n[INFO] Calculando máximos anuais para '{name_file}' com frequência: '{frequency}'")

    df = df.dropna()

    if frequency == 'hourly':
        required_cols = {'Year', 'Month', 'Day', 'Precipitation'}
        if not required_cols.issubset(df.columns):
            print(f"[ERRO] Colunas necessárias ausentes para frequência 'hourly': {required_cols}")
            return

        # Agrupar por data (Year, Month, Day) e somar a precipitação diária
        df_daily = df.groupby(['Year', 'Month', 'Day'], as_index=False)['Precipitation'].sum()
        print(f"[INFO] Dados horários agregados em {len(df_daily)} dias.")
    elif frequency == 'daily':
        df_daily = df.copy()
        if 'Month' not in df_daily.columns or 'Day' not in df_daily.columns:
            print("[WARNING] Colunas 'Month' e 'Day' ausentes nos dados diários. OK se não for necessário.")
    else:
        print("[ERRO] Frequência inválida. Use 'daily' ou 'hourly'.")
        return

    # Agrupar por ano e pegar o valor máximo
    df_max = df_daily.groupby('Year')['Precipitation'].max().reset_index()
    print(f"[INFO] Máximos anuais calculados para {df_max.shape[0]} anos.")
    
    # Remover outliers usando a função auxiliar caso especificado
    if not outliers:
        df_clean = remove_outliers_from_max(df_max)
        print(f"[INFO] Após remoção de outliers: {df_clean.shape[0]} anos restantes.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'max_daily_{name_file}.csv')
    df_clean.to_csv(output_path, index=False)

    print(f"[OK] Arquivo salvo em: {output_path}\n")
    return df_clean



def process_precipitation_series(
    dataframes: List[pd.DataFrame],
    frequency: Literal["daily", "hourly"] = "daily",
    plot: bool = True,
    return_fig: bool = False
) -> Tuple[pd.DataFrame, Optional[Tuple[plt.Figure, List[plt.Axes]]]]:
    """
    Processa séries temporais de precipitação a partir de DataFrames e retorna o DataFrame final,
    com opção de plotar os gráficos de dupla massa ou retorná-los.

    Parâmetros
    ----------
    dataframes : list[pd.DataFrame]
        Lista de DataFrames das estações.
    frequency : str
        Frequência dos dados ("daily" ou "hourly").
    plot : bool
        Se True, gera os gráficos com visualização padrão.
    return_fig : bool
        Se True, retorna fig e axes para customização posterior.

    Retorna
    -------
    df : pd.DataFrame
        DataFrame com séries unidas e colunas de precipitação acumulada.
    (fig, axes) : tuple, opcional
        Se return_fig=True, retorna objetos de plotagem.
    """

    verification_results = []
    processed_dfs: List[pd.DataFrame] = []

    # Verificação e preenchimento de dados
    for df in dataframes:
        result = verification(df, frequency=frequency)
        verification_results.append(result["status"])
        processed_dfs.append(df)

    complete_idx = [i for i, status in enumerate(verification_results) if status == 'complete']
    incomplete_idx = [i for i, status in enumerate(verification_results) if status == 'incomplete']

    if complete_idx:
        ref_idx = complete_idx[0]
        reference = processed_dfs[ref_idx]
        for i in incomplete_idx:
            processed_dfs[i] = fill_missing_data(processed_dfs[i], df_secondary=reference, frequency=frequency)
    elif incomplete_idx:
        for i in incomplete_idx:
            processed_dfs[i] = fill_missing_data(processed_dfs[i], frequency=frequency)

    # Unindo séries
    df = left_join_precipitation(*processed_dfs)
    df.columns = ['Date'] + [f"P_{i}" for i in range(len(processed_dfs))]
    df = df.dropna()
    df['P_average'] = df.iloc[:, 1:].mean(axis=1)

    # Cálculo acumulado
    for col in df.columns[1:]:
        df[f'Pacum_{col}'] = df[col].fillna(0).cumsum()

    fig, axes = None, []

    if plot or return_fig:
        sns.set_context("talk", font_scale=0.8)
        fig, axes = plt.subplots(1, len(processed_dfs), figsize=(6 * len(processed_dfs), 5), sharey=True)
        if len(processed_dfs) == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            sns.scatterplot(
                x="Pacum_P_average",
                y=f"Pacum_P_{i}",
                data=df,
                ax=ax,
                alpha=0.5,
                color='steelblue'
            )
            ax.set_xlabel("Média Pacum (mm)")
            ax.set_ylabel(f"Pacum Estação {i}")
            ax.set_title(f"Dupla Massa - Estação {i}")

        plt.tight_layout()
        if plot:
            plt.show()

    return (df, (fig, axes) if return_fig else None)