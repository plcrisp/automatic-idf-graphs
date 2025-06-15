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


class DisaggregationScenario(Enum):
    BASE = 'base'
    UMIDO = 'úmido'
    SECO = 'seco'



def aggregate_precipitation(df, interval, dt_min=False):
    """
    Agrega os dados de precipitação em intervalos especificados.

    Parâmetros:
    df (DataFrame): Um DataFrame contendo uma coluna 'Precipitation' 
                    com os dados de precipitação, indexados por tempo.
    interval (int): O intervalo de agregação desejado:
                    - Se `dt_min` for None, considera 'interval' em horas.
                    - Se `dt_min` for True, considera 'interval' em minutos.
    dt_min (int, opcional): A resolução temporal dos dados em minutos. 
                            Necessário se 'interval' for em minutos.

    Retorna:
    list: Uma lista contendo as somas de precipitação para cada intervalo 
          especificado.
    """
    
    # Verifica se o DataFrame contém a coluna 'Precipitation'
    if 'Precipitation' not in df.columns:
        raise ValueError("O DataFrame deve conter uma coluna 'Precipitation'.")
    
    # Remove valores ausentes da coluna 'Precipitation'
    df = df[['Precipitation']].dropna()
    
    # Lista para armazenar os resultados acumulados
    acum_list = []

    if not dt_min:
        # Caso padrão: agregação em horas
        n = interval  # Intervalo em horas
        for i in range(len(df) - n + 1):
            # Soma os valores de 'Precipitation' nas 'n' horas atuais
            acum = df.iloc[i:n + i]['Precipitation'].sum()
            acum_list.append(acum)
    
    else:
        # Caso em que o intervalo é em minutos
        n = interval // dt_min  # Calcula quantas entradas de dados devem ser somadas
        for i in range(len(df) - n + 1):
            # Soma os valores de 'Precipitation' nas 'n' entradas atuais
            acum = df.iloc[i:n + i]['Precipitation'].sum()
            acum_list.append(acum)

    return acum_list



def get_disagregation_factors(var_value, filename='parameters/fatores_desagregacao.csv'):
    """
    Lê os fatores de desagregação de um arquivo CSV e calcula fatores 
    baseados em um valor de variável fornecido.

    Parâmetros:
    var_value (float): Valor utilizado para calcular os fatores de desagregação.
    filename (str): Nome do arquivo CSV contendo os fatores de desagregação (padrão é 'fatores_desagregacao.csv').
    
    Retorna:
    DataFrame: Um DataFrame contendo os fatores de desagregação calculados.
    """
    # Lê o arquivo CSV contendo os fatores de desagregação
    df_disagreg_factors = pd.read_csv(filename)
    
    # Calcula os fatores de desagregação
    df_disagreg_factors['CETESB_p{v}'.format(v=var_value)] = df_disagreg_factors['CETESB_ger'] * (1 + var_value)
    df_disagreg_factors['CETESB_m{v}'.format(v=var_value)] = df_disagreg_factors['CETESB_ger'] * (1 - var_value)
    
    return df_disagreg_factors




def get_subdaily_from_disagregation_factors(df, scenario: DisaggregationScenario, var_value: float, name_file: str, directory='Results'):
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
    directory : str
        Diretório onde os resultados serão salvos.

    Retorna:
    -------
    None: Salva um CSV com os valores subdiários calculados.
    """
    df_subdaily = df.copy()
    df_disagreg_factors = get_disagregation_factors(var_value)

    if scenario == DisaggregationScenario.BASE:
        type_tag = 'ger'
    elif scenario == DisaggregationScenario.UMIDO:
        type_tag = f'p{var_value}'
    elif scenario == DisaggregationScenario.SECO:
        type_tag = f'm{var_value}'
    else:
        raise ValueError("Cenário inválido.")

    intervals = [5, 10, 15, 20, 25, 30, 60, 360, 480, 600, 720, 1440]
    col_name = f'CETESB_{type_tag}'

    if col_name not in df_disagreg_factors.columns:
        raise ValueError(f"Coluna {col_name} não encontrada em df_disagreg_factors.")

    for i, interval in enumerate(intervals):
        if i < len(df_disagreg_factors):
            factor = df_disagreg_factors[col_name].iloc[i]
            column_name = f'Max_{interval}min' if interval < 60 else f'Max_{interval // 60}h'
            df_subdaily[column_name] = df_subdaily['Precipitation'] * factor
        else:
            print(f"Intervalo {interval} não encontrado em fatores de desagregação.")

    output_path = f'{directory}/max_subdaily_{name_file}_{type_tag}.csv'
    df_subdaily.to_csv(output_path, index=False)
    print(f'Resultado salvo em {output_path}')


