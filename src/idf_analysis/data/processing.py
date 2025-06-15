"""
Este script realiza o processamento e análise de dados de precipitação.

Funcionalidades principais:
- Agregação dos dados em diferentes escalas temporais (anual, mensal, diária e horária).
- Salvamento dos dados agregados em arquivos CSV.
- Leitura dos arquivos CSV gerados.
- Visualização da distribuição dos dados de precipitação através de um gráfico de densidade.

Bibliotecas utilizadas:
- pandas: Manipulação de dados
- os: Gerenciamento de arquivos e diretórios
- matplotlib.pyplot: Visualização de gráficos
- seaborn: Gráficos estatísticos
"""

import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import date
from sklearn.ensemble import RandomForestRegressor



# Função para agregação flexível
def aggregate(df, vars):
    """
    Agrega os dados de precipitação com base nas colunas fornecidas.
    
    Parâmetros:
    df (DataFrame): O DataFrame com os dados.
    vars (list): Lista de variáveis para agrupar (ex: ['Year'], ['Year', 'Month']).
    
    Retorna:
    DataFrame: DataFrame com os dados agregados.
    """
    return df.groupby(vars).Precipitation.sum().reset_index()



# Função para salvar os arquivos CSV
def save_to_csv(df, name, var, directory):
    """
    Salva um DataFrame em formato CSV no diretório especificado.
    
    Parâmetros:
    df (DataFrame): O DataFrame a ser salvo.
    name (str): Nome base do arquivo.
    var (str): Nome da variável de agregação (ex: 'yearly', 'monthly').
    directory (str): Caminho do diretório onde salvar.
    """
    # Garante que o diretório existe
    os.makedirs(directory, exist_ok=True)
    
    # Define o caminho completo e salva o arquivo CSV
    file_path = os.path.join(directory, f'{name}_{var}.csv')
    df.to_csv(file_path, index=False)




# Função agregada e mais flexível para salvar diferentes agregações
def aggregate_to_csv(df, name, directory='Results'):
    """
    Agrega os dados e salva em arquivos CSV anuais, mensais, diários e por hora.
    
    Parâmetros:
    df (DataFrame): O DataFrame com os dados.
    name (str): Nome base para os arquivos.
    directory (str): Caminho do diretório onde salvar os resultados.
    """
    
    # Garante que o diretório existe
    Path(directory).mkdir(parents=True,exist_ok=True)
    
    # Agrega por ano
    df_yearly = aggregate(df, ['Year'])
    save_to_csv(df_yearly, name, 'yearly', directory)

    # Agrega por mês
    df_monthly = aggregate(df, ['Year', 'Month'])
    save_to_csv(df_monthly, name, 'monthly', directory)

    # Agrega por dia
    df_daily = aggregate(df, ['Year', 'Month', 'Day'])
    save_to_csv(df_daily, name, 'daily', directory)
    
    # Agrega por hora, caso a coluna exista
    if 'Hour' in df.columns:
        df_hourly = aggregate(df, ['Year', 'Month', 'Day', 'Hour'])
        save_to_csv(df_hourly, name, 'hourly', directory)




# Função para ler CSV
def read_csv(path):
    """
    Lê um arquivo CSV gerado pela agregação.
    
    Parâmetros:
    name (str): Nome base do arquivo.
    var (str): Variável de agregação (ex: 'yearly', 'monthly', 'daily', 'hourly', 'min').
    directory (str): Diretório onde os arquivos estão salvos.
    
    Retorna:
    DataFrame: O DataFrame lido do arquivo CSV.
    """
    file_path = os.path.join(path)
    return pd.read_csv(file_path)



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
        print("\n[INFO] DataFrame está vazio.\n")
        result["status"] = "empty"
        return result

    required_columns = {'Year', 'Month', 'Day'}
    if not required_columns.issubset(df.columns):
        print(f"\n[INFO] Colunas obrigatórias ausentes: {required_columns - set(df.columns)}\n")
        result["status"] = "missing_columns"
        return result

    # Cria coluna de data e ordena o DataFrame
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.sort_values('Date').reset_index(drop=True)

    d0 = df['Date'].iloc[0].date()
    di = df['Date'].iloc[-1].date()
    expected_days = (di - d0).days + 1
    actual_days = len(df)

    print(f"\n[INFO] Período da série: {d0} até {di}")
    print(f"[INFO] Dias esperados: {expected_days}")
    print(f"[INFO] Entradas no DataFrame: {actual_days}\n")

    missing_days = expected_days - actual_days

    if missing_days > 0:
        print(f"[WARNING] Série incompleta. Dias faltando: {missing_days}\n")
        result["status"] = "incomplete"
        result["missing_days"] = missing_days
    elif missing_days == 0:
        print("[OK] Série completa! Nenhum dia faltando.\n")
        result["status"] = "complete"
    else:
        print("[ERRO] Número de entradas excede o esperado. Verifique duplicatas ou erros.\n")
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



def fill_missing_data(path_main, path_secondary=None, overwrite=False):
    """
    Preenche os valores faltantes na coluna 'Precipitation' de um DataFrame.

    Se um segundo caminho for fornecido, tenta completar os dados faltantes
    com base em um modelo de Random Forest treinado com o segundo DataFrame.
    Caso contrário, aplica interpolação sazonal por mês.

    Se overwrite=True, substitui o arquivo original (path_main) com o novo DataFrame.

    Parâmetros:
    ----------
    path_main : str
        Caminho para o CSV principal com possíveis valores faltantes.
    path_secondary : str, opcional
        Caminho para um segundo CSV com dados completos ou com menos faltas.
    overwrite : bool, opcional (padrão: False)
        Se True, sobrescreve o arquivo path_main com os dados preenchidos.

    Retorna:
    -------
    df_main : pandas.DataFrame
        DataFrame com a coluna 'Precipitation' preenchida.
    """
    df_main = set_date(read_csv(path_main))

    if path_secondary:
        df_secondary = set_date(read_csv(path_secondary))

        for df in (df_main, df_secondary):
            if not {'Year', 'Month', 'Day', 'Precipitation'}.issubset(df.columns):
                raise ValueError("Os DataFrames devem conter as colunas: Year, Month, Day, Precipitation")

        df_main['Key'] = df_main[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1)
        df_secondary['Key'] = df_secondary[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1)

        merged = df_main[['Key', 'Precipitation']].merge(
            df_secondary[['Key', 'Precipitation']],
            on='Key',
            how='left',
            suffixes=('_main', '_sec')
        )

        valid = merged.dropna(subset=['Precipitation_main', 'Precipitation_sec'])

        if len(valid) >= 10:  # Apenas treina se tiver dados suficientes
            valid = valid.copy()
            valid['Year'] = pd.to_datetime(valid['Key']).dt.year
            valid['Month'] = pd.to_datetime(valid['Key']).dt.month
            valid['Day'] = pd.to_datetime(valid['Key']).dt.day

            X_train = valid[['Precipitation_sec', 'Year', 'Month', 'Day']]
            y_train = valid['Precipitation_main']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            # Prepara os dados ausentes para previsão
            to_predict = merged[merged['Precipitation_main'].isna() & merged['Precipitation_sec'].notna()].copy()
            to_predict['Year'] = pd.to_datetime(to_predict['Key']).dt.year
            to_predict['Month'] = pd.to_datetime(to_predict['Key']).dt.month
            to_predict['Day'] = pd.to_datetime(to_predict['Key']).dt.day

            X_pred = to_predict[['Precipitation_sec', 'Year', 'Month', 'Day']]
            merged.loc[to_predict.index, 'Precipitation_main'] = model.predict(X_pred)

        else:
            print("Poucos dados para treinar Random Forest. Usando interpolação sazonal.")
            interpolated = (
                df_main.groupby('Month')['Precipitation']
                .apply(lambda group: group.interpolate(method='linear'))
            )
            df_main['Precipitation'] = interpolated.reset_index(level=0, drop=True)
            return df_main

        # Atualiza os valores de precipitação no df_main
        df_main.set_index('Key', inplace=True)
        merged.set_index('Key', inplace=True)
        df_main['Precipitation'] = merged['Precipitation_main']
        df_main.reset_index(drop=True, inplace=True)

    else:
        interpolated = (
            df_main.groupby('Month')['Precipitation']
            .apply(lambda group: group.interpolate(method='linear'))
        )
        df_main['Precipitation'] = interpolated.reset_index(level=0, drop=True)

    if overwrite:
        df_main.to_csv(path_main, index=False)

        filename = os.path.basename(path_main)
        directory = os.path.dirname(path_main)
        name = os.path.splitext(filename)[0]
        name = re.sub(r'_(daily|yearly|monthly|hourly)$', '', name)

        aggregate_to_csv(df=df_main, name=name, directory=directory)

    return df_main



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
