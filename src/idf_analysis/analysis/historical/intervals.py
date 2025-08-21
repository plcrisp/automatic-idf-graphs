"""
Este script contém funções para processamento e análise de dados de precipitação, incluindo agregação temporal e desagregação subdiária. As principais funcionalidades são:

1. **Agregação de Precipitação (`aggregate_precipitation`)**:
   - Soma os valores de precipitação em intervalos de tempo especificados.
   - Permite agregação por horas ou minutos, dependendo da resolução dos dados.
   - Remove valores ausentes para garantir cálculos consistentes.

2. **Cálculo de Fatores de Desagregação (`get_disagregation_factors`)**:
   - Lê um arquivo CSV contendo fatores de desagregação pré-definidos.
   - Calcula variações dos fatores para cenários de aumento e redução baseados em um valor fornecido pelo usuário.

3. **Cálculo de Precipitação Subdiária (`get_subdaily_from_disagregation_factors`)**:
   - Aplica fatores de desagregação para estimar precipitação em diferentes intervalos subdiários.
   - Suporta três tipos de desagregação: original, aumentada ('plus') e reduzida ('minus').
   - Gera e salva um arquivo CSV com os valores desagregados para diferentes intervalos (de 5 minutos até 24 horas).

Este código é útil para análise hidrológica, auxiliando na preparação de dados de precipitação para modelagem e estudos de variabilidade temporal.
"""

import pandas as pd
from enum import Enum
from importlib import resources
from typing import Literal
from pathlib import Path



class DisaggregationScenario(Enum):
    BASE = 'base'
    UMIDO = 'úmido'
    SECO = 'seco'



def aggregate_precipitation(df, interval, dt_min=False):
    """
    Agrega os dados de precipitação em janelas móveis reais (rolling),
    respeitando a data/hora, com intervalos em horas ou minutos.

    Parâmetros:
    - df (DataFrame): Deve conter colunas 'Year', 'Month', 'Day', 'Hour' e opcionalmente 'Minute'.
    - interval (int): Intervalo desejado de agregação (em horas ou minutos).
    - dt_min (int or False): Resolução temporal dos dados (ex: 5, 10, 60). 
                             Se False, assume dados horários (60 min).

    Retorna:
    - List[float]: Lista de totais de precipitação por janela móvel.
    """
    
    df = df.copy()

    # Verificação básica
    if 'Precipitation' not in df.columns:
        raise ValueError("O DataFrame deve conter a coluna 'Precipitation'.")

    # Construir índice temporal
    if dt_min:
        if 'Minute' not in df.columns:
            df['Minute'] = 0
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    else:
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']]) + pd.to_timedelta(df['Hour'], unit='h')

    df = df[['Date', 'Precipitation']].dropna().sort_values('Date').set_index('Date')

    # Tamanho da janela em unidades temporais (ex: 6 valores para 6h se dt_min=False)
    if not dt_min:
        window_size = interval  # horas
        freq = 'h'
    else:
        window_size = interval // dt_min
        freq = f'{dt_min}min'

    # Verifica se o índice é regular (apenas avisa)
    expected = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    if len(expected) != len(df):
        print(f"[WARNING] Série irregular ou com buracos: {len(df)} observações vs {len(expected)} esperadas")

    # Rolling sum com janela deslizante real
    rolling_sum = df['Precipitation'].rolling(window=window_size, min_periods=window_size).sum().dropna()

    return rolling_sum.tolist()



def get_disaggregation_factors(var_value: float) -> pd.DataFrame:
    """
    Lê os fatores de desagregação de um arquivo de recursos do pacote e 
    calcula fatores baseados em um valor de variável fornecido.

    Parâmetros:
        var_value (float): Valor utilizado para calcular os fatores de desagregação.
    
    Retorna:
        pd.DataFrame: Um DataFrame contendo os fatores de desagregação calculados.
                      Retorna um DataFrame vazio se o arquivo não for encontrado.
    """
    try:
        with resources.files('idf_analysis.resources').joinpath('disaggregation_factors.csv').open('r', encoding='utf-8') as f:
            
            df_disagreg_factors = pd.read_csv(f)
            
        df_disagreg_factors[f'CETESB_p{var_value}'] = df_disagreg_factors['CETESB_ger'] * (1 + var_value)
        df_disagreg_factors[f'CETESB_m{var_value}'] = df_disagreg_factors['CETESB_ger'] * (1 - var_value)
        
        return df_disagreg_factors

    except (FileNotFoundError, ModuleNotFoundError):
        print("Erro: Não foi possível encontrar o arquivo 'fatores_desagregacao.csv' dentro do pacote.")
        return pd.DataFrame()




def get_subdaily_from_disaggregation_factors(
    df,
    scenario: DisaggregationScenario,
    name_file: str,
    var_value: float = 0.2,
    output_dir='Results',
    frequency: Literal["daily", "hourly"] = "daily",
):
    """
    Calcula os valores subdiários de precipitação baseados em fatores de desagregação.

    Parâmetros:
    ----------
    df : DataFrame
        DataFrame contendo os dados de precipitação.
    scenario : DisaggregationScenario
        Enum definindo o tipo de cenário de desagregação:
            - DisaggregationScenario.BASE
            - DisaggregationScenario.UMIDO
            - DisaggregationScenario.SECO
    var_value : float
        Valor usado para ajustar os cenários úmido e seco (ex.: 0.1, 0.2, 0.3).
    name_file : str
        Nome base do arquivo a ser salvo (sem extensão).
    output_dir : str
        Diretório onde os resultados serão salvos.
    frequency : str
        "daily" para usar coluna 'Precipitation' e intervalos minutais (< 60 min),
        "subdaily" para usar coluna 'Max_24h' e incluir todos os intervalos (minutais e horários).

    Retorna:
    -------
    DataFrame: O DataFrame com colunas desagregadas adicionadas.
    """
    df_subdaily = df.copy()
    df_disagreg_factors = get_disaggregation_factors(var_value)

    if scenario == DisaggregationScenario.BASE:
        type_tag = 'ger'
    elif scenario == DisaggregationScenario.UMIDO:
        type_tag = f'p{var_value}'
    elif scenario == DisaggregationScenario.SECO:
        type_tag = f'm{var_value}'
    else:
        raise ValueError("Cenário inválido.")
    
    intervals = [5, 10, 15, 20, 25, 30, 60, 360, 480, 600, 720, 1440]  # Todos

    # Definir coluna de referência baseada na frequência
    if frequency == "daily":
        reference_col = 'Precipitation'
    elif frequency == "hourly":
        reference_col = 'Max_24h'
        
    else:
        raise ValueError("Frequência inválida. Use 'daily' ou 'hourly'.")

    # Verificar se a coluna de referência existe
    if reference_col not in df_subdaily.columns:
        raise ValueError(f"Coluna '{reference_col}' não encontrada no DataFrame para frequência '{frequency}'.")

    col_name = f'CETESB_{type_tag}'
    if col_name not in df_disagreg_factors.columns:
        raise ValueError(f"Coluna {col_name} não encontrada em df_disagreg_factors.")

    for i, interval in enumerate(intervals):
        if i < len(df_disagreg_factors):
            factor = df_disagreg_factors[col_name].iloc[i]
            column_name = f'Max_{interval}min' if interval < 60 else f'Max_{interval // 60}h'
            df_subdaily[column_name] = (df_subdaily[reference_col] * factor).round(2)
        else:
            print(f"[WARNING] Intervalo {interval} não encontrado em fatores de desagregação.")

    output_path = Path(output_dir) / f"max_daily_{name_file}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_subdaily.to_csv(output_path, index=False)
    print(f'[OK] Resultado salvo em: {output_path}')
    return df_subdaily


