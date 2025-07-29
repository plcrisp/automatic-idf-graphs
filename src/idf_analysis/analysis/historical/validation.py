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



def max_annual_precipitation(df, name_file, output_dir='Results', frequency: Literal['daily', 'hourly'] = 'daily'):
    """
    Calcula o valor máximo de precipitação anual e remove outliers.
    Para dados horários, soma a precipitação por dia antes de calcular os máximos.

    Parâmetros:
    - df (DataFrame): Deve conter colunas 'Year', 'Precipitation', e dependendo da frequência: 'Month', 'Day', 'Hour'.
    - name_file (str): Nome base do arquivo de saída (sem extensão).
    - output_dir (str): Diretório onde o CSV será salvo.
    - frequency (str): 'daily' (espera valores diários) ou 'hourly' (soma por dia antes de agrupar por ano).

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

    # Remoção de outliers
    df_clean = remove_outliers_from_max(df_max)
    print(f"[INFO] Após remoção de outliers: {df_clean.shape[0]} anos restantes.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'max_daily_{name_file}.csv')
    df_clean.to_csv(output_path, index=False)

    print(f"[OK] Arquivo salvo em: {output_path}\n")
    return df_clean



def process_precipitation_series(
    file_names: List[str],
    frequency: Literal["daily", "hourly"] = "daily",
    plot: bool = True,
    return_fig: bool = False
) -> Tuple[pd.DataFrame, Optional[Tuple[plt.Figure, List[plt.Axes]]]]:
    """
    Processa séries temporais de precipitação e retorna o DataFrame final,
    com opção de plotar os gráficos de dupla massa ou retorná-los.

    Parâmetros:
    - file_names: Lista com os nomes dos arquivos CSV.
    - frequency: Frequência dos dados ("daily" ou "hourly").
    - plot: Se True, gera os gráficos com visualização padrão.
    - return_fig: Se True, retorna fig e axes para customização posterior.

    Retorna:
    - df: DataFrame com séries unidas e colunas de precipitação acumulada.
    - (fig, axes): Se return_fig=True, retorna objetos de plotagem.
    """

    def load_and_verify(file_name):
        print(f"[INFO] Lendo e verificando: {file_name}")
        df = read_csv(file_name)
        result = verification(df, frequency=frequency)
        return df, result["status"]

    verification_results = {}
    dataframes: Dict[str, pd.DataFrame] = {}

    for name in file_names:
        df, status = load_and_verify(name)
        dataframes[name] = df
        verification_results[name] = status

    complete_paths = [name for name, status in verification_results.items() if status == 'complete']
    incomplete_paths = [name for name, status in verification_results.items() if status == 'incomplete']

    if complete_paths:
        reference = complete_paths[0]
        print(f"[INFO] Usando '{reference}' como referência para preenchimento.")
        for name in incomplete_paths:
            print(f"[INFO] Preenchendo '{name}' com base em '{reference}'...")
            dataframes[name] = fill_missing_data(path_main=name, path_secondary=reference, frequency=frequency, overwrite=False)
    elif incomplete_paths:
        print("[INFO] Nenhum dataset completo encontrado. Preenchendo todos individualmente...")
        for name in incomplete_paths:
            print(f"[INFO] Preenchendo '{name}' individualmente...")
            dataframes[name] = fill_missing_data(path_main=name, frequency=frequency)
    else:
        print("[INFO] Todos os datasets estão completos.")

    print(f"[INFO] Unindo séries e calculando média {'horária' if frequency == 'hourly' else 'diária'}...")

    df = left_join_precipitation(*dataframes.values())
    df.columns = ['Date'] + [f"P_{i}" for i in range(len(file_names))]  # P_0, P_1, ...
    df = df.dropna()
    df['P_average'] = df.iloc[:, 1:].mean(axis=1)

    for col in df.columns[1:]:
        df[f'Pacum_{col}'] = df[col].fillna(0).cumsum()

    fig, axes = None, []

    if plot or return_fig:
        print("[INFO] Gerando gráficos de dupla massa...")
        sns.set_context("talk", font_scale=0.8)
        fig, axes = plt.subplots(1, len(file_names), figsize=(6 * len(file_names), 5), sharey=True)

        if len(file_names) == 1:
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