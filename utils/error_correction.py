"""
Este script realiza a verificação e o processamento de séries temporais de dados meteorológicos.
Inclui funções para verificar a integridade da série temporal, criar uma coluna de data como índice,
preencher valores ausentes por interpolação sazonal e remover outliers usando o método do IQR.
Utiliza a biblioteca Pandas para manipulação de dados e uma biblioteca auxiliar para leitura de CSV.
"""

import pandas as pd
from datetime import date
from .data_processing import read_csv

def verification(df):
    """
    Verifica a integridade de uma série temporal de dados meteorológicos.

    Parâmetros:
        df (DataFrame): Um DataFrame contendo colunas 'Year', 'Month', 'Day'.

    A função compara o número de dias consecutivos entre a primeira e última data
    com o número de registros na base. Informa se há lacunas ou se a série está completa.
    
    Retorna:
        dict: Um dicionário com o status da verificação e o número de dias faltantes (se houver).
    """

    result = {"status": "", "missing_days": 0}

    if df.empty:
        print("[INFO] DataFrame está vazio.")
        result["status"] = "empty"
        return result

    required_columns = {'Year', 'Month', 'Day'}
    if not required_columns.issubset(df.columns):
        print(f"[INFO] Colunas obrigatórias ausentes: {required_columns - set(df.columns)}")
        result["status"] = "missing_columns"
        return result

    # Cria coluna de data e ordena o DataFrame
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.sort_values('Date').reset_index(drop=True)

    d0 = df['Date'].iloc[0].date()
    di = df['Date'].iloc[-1].date()
    expected_days = (di - d0).days + 1
    actual_days = len(df)

    print(f"[INFO] Período da série: {d0} até {di}")
    print(f"[INFO] Dias esperados: {expected_days}")
    print(f"[INFO] Entradas no DataFrame: {actual_days}")

    missing_days = expected_days - actual_days

    if missing_days > 0:
        print(f"[WARNING] Série incompleta. Dias faltando: {missing_days}")
        result["status"] = "incomplete"
        result["missing_days"] = missing_days
    elif missing_days == 0:
        print("[OK] Série completa! Nenhum dia faltando.")
        result["status"] = "complete"
    else:
        print("[ERRO] Número de entradas excede o esperado. Verifique duplicatas ou erros.")
        result["status"] = "invalid_dataset"

    return result
        
        
        
def set_date(df):
    """
    Cria uma coluna 'Date' a partir das colunas 'Year', 'Month' e 'Day', define-a como índice e retorna o DataFrame.

    Parâmetros:
    df (DataFrame): DataFrame contendo as colunas 'Year', 'Month' e 'Day'.

    Retorna:
    DataFrame: DataFrame atualizado com a nova coluna 'Date' e o índice configurado.
    """
    # Cria a coluna 'Date' combinando 'Year', 'Month' e 'Day', ignorando erros em datas inválidas
    df['Date'] = [date(y, m, d) if pd.notnull(y) and pd.notnull(m) and pd.notnull(d) else pd.NaT
                  for y, m, d in zip(df['Year'], df['Month'], df['Day'])]

    # Define 'Date' como índice e retorna o DataFrame atualizado
    df.set_index('Date', inplace=True)
    
    # Cria uma faixa de datas completa entre a primeira e a última data usando o índice
    idx = pd.date_range(df.index[0], df.index[-1])
    
    # Reindexa o DataFrame para preencher as datas faltantes e recria a coluna 'Date'
    df = df.reindex(idx)
    df['Date'] = df.index

    # Preenche as colunas 'Year', 'Month' e 'Day' com os valores corretos
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day

    return df



def fill_missing_data(name, var):
    """
    Preenche os valores faltantes na coluna 'Precipitation' de um DataFrame
    utilizando interpolação sazonal (baseada em grupos mensais).

    Parâmetros:
    ----------
    name : str
        Nome do arquivo ou base de dados a ser carregado.
    var : str
        Tipo de dados ou variável a ser processada (ex.: 'daily').

    Retorna:
    -------
    df : pandas.DataFrame
        DataFrame com os valores interpolados na coluna 'Precipitation'.
        Os índices do DataFrame permanecem alinhados com os valores originais.
    """
    df = set_date(read_csv(name, var))
    
    # Realiza a interpolação sazonal (por mês)
    interpolated = (
        df.groupby('Month')['Precipitation']
        .apply(lambda group: group.interpolate(method='linear'))
    )

    # Realinha os índices do resultado interpolado com o DataFrame original
    df['Precipitation'] = interpolated.reset_index(level=0, drop=True)

    return df



def remove_outliers_from_max(df, column='Precipitation', duration=0):
    """
    Remove outliers da coluna 'Precipitation' de um DataFrame, sem agrupar os dados.

    Args:
        df (pd.DataFrame): DataFrame com coluna 'Precipitation'.
        duration (int, optional): Duração usada para renomear a coluna (padrão: 0, sem renomeação).

    Returns:
        pd.DataFrame: DataFrame filtrado sem outliers na coluna 'Precipitation'.
    """
    # Removendo valores nulos
    df_no_na = df.dropna()

    # Calcula os limites para remoção de outliers usando o IQR
    q1 = df_no_na[column].quantile(0.25)
    q3 = df_no_na[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filtra os dados para remover os outliers
    df_filtered = df_no_na[(df_no_na[column] > lower_bound) & 
                           (df_no_na[column] < upper_bound)]
    
    # Caso uma duração seja passada, renomeia a coluna
    if duration != 0:
        df_filtered.columns = ['Max_{dur}'.format(dur=duration)]
    
    return df_filtered
