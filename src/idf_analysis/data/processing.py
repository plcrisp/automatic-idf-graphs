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

from typing import Literal
from pathlib import Path
from datetime import date, datetime
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



def verification(df: pd.DataFrame, frequency: Literal["yearly", "monthly", "daily", "hourly"] = "daily"):
    """
    Verifica a integridade de uma série temporal de dados meteorológicos.

    Parâmetros:
        df (DataFrame): Deve conter colunas correspondentes à frequência analisada.
        frequency (str): Frequência da verificação ('yearly', 'monthly', 'daily', 'hourly').

    Retorna:
        dict: Status da verificação e número de períodos faltantes (se houver).
    """
    result = {"status": "", "missing": 0}

    if df.empty:
        print("\n[INFO] DataFrame está vazio.\n")
        result["status"] = "empty"
        return result

    if frequency == "yearly":
        required_columns = {"Year"}
    elif frequency == "monthly":
        required_columns = {"Year", "Month"}
    elif frequency == "daily":
        required_columns = {"Year", "Month", "Day"}
    elif frequency == "hourly":
        required_columns = {"Year", "Month", "Day", "Hour"}
    else:
        raise ValueError("Frequência inválida. Use: 'yearly', 'monthly', 'daily' ou 'hourly'.")

    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        print(f"\n[INFO] Colunas obrigatórias ausentes: {missing_cols}\n")
        result["status"] = "missing_columns"
        return result

    # Criação da coluna de tempo base
    if frequency == "yearly":
        df["Date"] = pd.to_datetime(df["Year"], format="%Y")
    elif frequency == "monthly":
        df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(Day=1))
    elif frequency == "daily":
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]])
    elif frequency == "hourly":
        df["Date"] = pd.to_datetime(df[["Year", "Month", "Day"]]) + pd.to_timedelta(df["Hour"], unit="h")

    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

    d0 = df["Date"].iloc[0]
    di = df["Date"].iloc[-1]

    print(f"\n[INFO] Período da série: {d0} até {di}")

    # Gera índice esperado de datas
    if frequency == "yearly":
        expected_index = pd.date_range(d0, di, freq="YS")
    elif frequency == "monthly":
        expected_index = pd.date_range(d0, di, freq="MS")
    elif frequency == "daily":
        expected_index = pd.date_range(d0, di, freq="D")
    elif frequency == "hourly":
        expected_index = pd.date_range(d0, di, freq="h")

    expected_count = len(expected_index)
    actual_count = len(df)

    print(f"[INFO] Períodos esperados: {expected_count}")
    print(f"[INFO] Entradas no DataFrame: {actual_count}\n")

    missing = expected_count - actual_count

    if missing > 0:
        print(f"[WARNING] Série incompleta. Períodos faltando: {missing}\n")
        result["status"] = "incomplete"
        result["missing"] = missing
    elif missing == 0:
        print("[OK] Série completa! Nenhum período faltando.\n")
        result["status"] = "complete"
    else:
        print("[ERRO] Número de entradas excede o esperado. Verifique duplicatas ou erros.\n")
        result["status"] = "invalid_dataset"

    return result
        
        
        
def set_date(df):
    """
    Cria uma coluna 'Date' a partir de 'Year', 'Month', 'Day' (e opcionalmente 'Hour'),
    define-a como índice, e preenche intervalos de tempo ausentes com base na resolução.

    Parâmetros:
    ----------
    df : pandas.DataFrame
        DataFrame contendo as colunas 'Year', 'Month', 'Day' e opcionalmente 'Hour'.

    Retorna:
    -------
    df : pandas.DataFrame
        DataFrame atualizado com índice datetime e colunas reconstruídas.
    """
    has_hour = 'Hour' in df.columns

    # Criação segura do datetime
    if has_hour:
        df['Date'] = [
            datetime(y, m, d, h) if pd.notnull(y) and pd.notnull(m) and pd.notnull(d) and pd.notnull(h)
            else pd.NaT
            for y, m, d, h in zip(df['Year'], df['Month'], df['Day'], df['Hour'])
        ]
    else:
        df['Date'] = [
            date(y, m, d) if pd.notnull(y) and pd.notnull(m) and pd.notnull(d)
            else pd.NaT
            for y, m, d in zip(df['Year'], df['Month'], df['Day'])
        ]

    df.set_index('Date', inplace=True)

    # Gera índice contínuo com base na frequência
    start, end = df.index.min(), df.index.max()
    freq = 'h' if has_hour else 'D'
    full_idx = pd.date_range(start=start, end=end, freq=freq)

    # Reindexa e reconstrói as colunas
    df = df.reindex(full_idx)
    df['Date'] = df.index
    df['Year'] = df.index.year
    df['Month'] = df.index.month
    df['Day'] = df.index.day
    if has_hour:
        df['Hour'] = df.index.hour

    return df



def interpolate_by_frequency(df, frequency: Literal["yearly", "monthly", "daily", "hourly"] = "daily"):
    """
    Aplica interpolação em uma série de acordo com a frequência desejada.
    
    Parâmetros:
        df (DataFrame): DataFrame com colunas 'Precipitation', 'Year', 'Month', 'Day', e opcionalmente 'Hour'.
        frequency (str): 'yearly', 'monthly', 'daily' ou 'hourly'.
    
    Retorna:
        Series: Série interpolada.
    """
    if frequency == 'yearly':
        return df.groupby('Year')['Precipitation'].transform(lambda x: x.interpolate(method='linear'))

    elif frequency == 'monthly':
        return df.groupby(['Year', 'Month'])['Precipitation'].transform(lambda x: x.interpolate(method='linear'))

    elif frequency == 'daily':
        return df.groupby('Month')['Precipitation'].transform(lambda x: x.interpolate(method='linear'))

    elif frequency == 'hourly':
        if 'Hour' not in df.columns:
            raise ValueError("Coluna 'Hour' necessária para interpolação horária.")
        return df.groupby(['Year', 'Month', 'Day'])['Precipitation'].transform(lambda x: x.interpolate(method='linear'))

    else:
        raise ValueError("Frequência inválida. Use 'yearly', 'monthly', 'daily' ou 'hourly'.")



def fill_missing_data(path_main, path_secondary=None, overwrite=False, frequency: Literal["yearly", "monthly", "daily", "hourly"] = "daily"):
    """
    Preenche os valores faltantes na coluna 'Precipitation' de um DataFrame.

    Usa interpolação por frequência ou Random Forest com base no path_secondary.

    Parâmetros:
    ----------
    path_main : str
        Caminho para o CSV principal com valores faltantes.
    path_secondary : str, opcional
        Caminho para um segundo CSV com dados mais completos.
    overwrite : bool, opcional
        Se True, sobrescreve o CSV original.
    frequency : str
        Uma entre: 'yearly', 'monthly', 'daily', 'hourly'.

    Retorna:
    -------
    df_main : pandas.DataFrame
        DataFrame com a coluna 'Precipitation' preenchida.
    """
    df_main = set_date(read_csv(path_main))

    if path_secondary:
        df_secondary = set_date(read_csv(path_secondary))

        # Verificações mínimas
        required = {'Year', 'Month', 'Day', 'Precipitation'}
        if not required.issubset(df_main.columns) or not required.issubset(df_secondary.columns):
            raise ValueError(f"Colunas obrigatórias ausentes: {required}")

        has_hour = frequency == 'hourly' and 'Hour' in df_main.columns and 'Hour' in df_secondary.columns

        # Cria chave única para merge
        if has_hour:
            df_main['Key'] = pd.to_datetime(df_main[['Year', 'Month', 'Day', 'Hour']]).dt.strftime('%Y-%m-%d %H:%M')
            df_secondary['Key'] = pd.to_datetime(df_secondary[['Year', 'Month', 'Day', 'Hour']]).dt.strftime('%Y-%m-%d %H:%M')
        else:
            df_main['Key'] = pd.to_datetime(df_main[['Year', 'Month', 'Day']]).dt.strftime('%Y-%m-%d')
            df_secondary['Key'] = pd.to_datetime(df_secondary[['Year', 'Month', 'Day']]).dt.strftime('%Y-%m-%d')

        merged = df_main[['Key', 'Precipitation']].merge(
            df_secondary[['Key', 'Precipitation']],
            on='Key',
            how='left',
            suffixes=('_main', '_sec')
        )

        valid = merged.dropna(subset=['Precipitation_main', 'Precipitation_sec'])

        if len(valid) >= 10:
            valid = valid.copy()
            valid['Year'] = pd.to_datetime(valid['Key']).dt.year
            valid['Month'] = pd.to_datetime(valid['Key']).dt.month
            valid['Day'] = pd.to_datetime(valid['Key']).dt.day
            if has_hour:
                valid['Hour'] = pd.to_datetime(valid['Key']).dt.hour

            feature_cols = ['Precipitation_sec', 'Year', 'Month', 'Day']
            if has_hour:
                feature_cols.append('Hour')

            X_train = valid[feature_cols]
            y_train = valid['Precipitation_main']

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            to_predict = merged[merged['Precipitation_main'].isna() & merged['Precipitation_sec'].notna()].copy()
            to_predict['Year'] = pd.to_datetime(to_predict['Key']).dt.year
            to_predict['Month'] = pd.to_datetime(to_predict['Key']).dt.month
            to_predict['Day'] = pd.to_datetime(to_predict['Key']).dt.day
            if has_hour:
                to_predict['Hour'] = pd.to_datetime(to_predict['Key']).dt.hour

            X_pred = to_predict[feature_cols]
            merged.loc[to_predict.index, 'Precipitation_main'] = model.predict(X_pred)

        else:
            print("Poucos dados para treinar Random Forest. Usando interpolação.")
            interpolated = interpolate_by_frequency(df_main, frequency)
            df_main['Precipitation'] = interpolated
            return df_main

        df_main.set_index('Key', inplace=True)
        merged.set_index('Key', inplace=True)
        df_main['Precipitation'] = merged['Precipitation_main']
        df_main.reset_index(drop=True, inplace=True)

    else:
        interpolated = interpolate_by_frequency(df_main, frequency)
        df_main['Precipitation'] = interpolated

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
