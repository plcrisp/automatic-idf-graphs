import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from datetime import datetime, date

def set_date(df):
    """
    Padroniza DataFrame de precipitação:
    - Cria coluna 'Date' a partir de Year/Month/Day/(Hour)
    - Agrega por data somando 'Precipitation'
    """
    has_hour = 'Hour' in df.columns

    # Criação do datetime seguro
    if has_hour:
        df['Date'] = pd.to_datetime(
            dict(year=df['Year'], month=df['Month'], day=df['Day'], hour=df['Hour']),
            errors='coerce'
        )
    else:
        df['Date'] = pd.to_datetime(
            dict(year=df['Year'], month=df['Month'], day=df['Day']),
            errors='coerce'
        )

    # Agrupa por data somando precipitação
    if 'Precipitation' in df.columns:
        df = df.groupby('Date', as_index=False)['Precipitation'].sum()
    else:
        df = df.groupby('Date', as_index=False).first()

    return df


def left_join_precipitation(*dfs):
    """
    Combina múltiplos DataFrames de precipitação em um único DataFrame
    com base na data, padronizando nomes de colunas.
    """
    result_df = set_date(dfs[0]).rename(columns={'Precipitation': 'P1'})

    for i, df in enumerate(dfs[1:], start=2):
        df_temp = set_date(df).rename(columns={'Precipitation': f'P{i}'})
        result_df = result_df.merge(df_temp, on='Date', how='inner')

    return result_df


def correlation_plots(*dfs, sample_max=5000, log_transform=True, add_regression=True):
    """
    Gera pairplots e calcula correlação entre séries de precipitação.
    
    Melhorias:
    - Agregação automática diária
    - Pairplot com log opcional e linha de regressão
    - Amostragem para datasets grandes
    """
    # Junta dados de todas as estações
    df = left_join_precipitation(*dfs)
    df = df.set_index('Date')

    # Detecta dados horários e agrega
    freq = pd.infer_freq(df.index[:10])
    if freq and 'H' in freq.upper():
        print("[INFO] Dados horários detectados → agregando para diário...")
        df = df.resample('D').sum()

    df = df.dropna(how='all')

    # Aplica log1p para visualização
    df_plot = df.copy()
    if log_transform:
        df_plot = np.log1p(df_plot)

    # Amostragem para pairplot
    if len(df_plot) > sample_max:
        df_plot = df_plot.sample(n=sample_max, random_state=42)

    print(f"[INFO] Gerando pairplot com {len(df_plot)} pontos...")
    kind = 'reg' if add_regression else 'scatter'
    sns.pairplot(df_plot, kind=kind, plot_kws={'scatter_kws': {'s': 15, 'alpha': 0.6}})
    plt.show()

    # Matriz de correlação
    corr_pearson = df.corr(method='pearson')

    # Matriz de p-valores
    pvalues_pearson = pd.DataFrame(np.ones_like(corr_pearson), columns=df.columns, index=df.columns)
    cols = df.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r, p = pearsonr(df[cols[i]], df[cols[j]])
            pvalues_pearson.loc[cols[i], cols[j]] = p
            pvalues_pearson.loc[cols[j], cols[i]] = p

    print("\n[INFO] ----- Resultados da Correlação -----\n")
    print("Matriz de Correlação (r de Pearson):")
    print(corr_pearson.to_string(float_format="%.4f"))
    print("\nMatriz de P-valores:")
    print(pvalues_pearson.to_string(float_format="%.4e"))

    return corr_pearson, pvalues_pearson