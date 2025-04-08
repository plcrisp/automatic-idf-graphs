"""
Este script é uma ferramenta completa para análise de precipitação extrema, com foco em cálculos estatísticos e análise de extremos subdiários.  
As funções estão organizadas em duas categorias principais:

1. Cálculo Estatístico de Extremos:
   - calculate_p90: Calcula o percentil de 90% (P90) para valores de precipitação e gera o gráfico da probabilidade acumulada de não excedência.
   - max_annual_precipitation: Determina a precipitação máxima anual, remove outliers e salva os resultados em CSV.

2. Análise de Dados Subdiários:
   - get_subdaily_extremes: Calcula os valores máximos e mínimos de precipitação acumulada em intervalos horários ou de minutos para cada ano.
   - get_max_subdaily_table: Gera tabelas com os máximos subdiários para múltiplos intervalos e salva os resultados.
   - merge_max_tables: Mescla as tabelas de máximos de precipitação em diferentes intervalos (horas e minutos) em um único arquivo consolidado.

Cada função já está preparada para tratamento de outliers, manipulação de diretórios e exportação automática para CSV, facilitando a análise climática.
"""

import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from .error_correction import remove_outliers_from_max
from .intervals_manipulation import aggregate_precipitation


"""
--------------------------------------------------------------------------------------------------------------
----------------------------------------- CÁLCULO ESTATÍSTICO DE EXTREMOS ------------------------------------
--------------------------------------------------------------------------------------------------------------
"""


def calculate_p90(df):
    """
    Calcula o percentil de 90% (P90) para valores de precipitação, ou seja, o valor que é excedido em apenas 10% das observações.
    Também plota o gráfico da probabilidade acumulada de não excedência em função da precipitação.

    Parâmetros:
    df (pd.DataFrame): DataFrame com uma coluna 'Precipitation' contendo valores de precipitação.

    Retorna:
    float: O valor de precipitação correspondente ao percentil de 90% (P90).
    """
    
    # Filtra e ordena os valores de precipitação, excluindo zeros
    df = df[['Precipitation']].query('Precipitation != 0').sort_values('Precipitation').reset_index(drop=True)

    # Calcula a probabilidade de não excedência para cada valor em porcentagem
    df['Probability'] = (df.index + 1) / len(df) * 100
    
    # Filtra o valor de precipitação onde a probabilidade de não excedência é aproximadamente 90%
    p90_value = df.loc[df['Probability'] >= 90, 'Precipitation'].iloc[0]

    # Plota o gráfico da probabilidade acumulada de não excedência
    sns.lineplot(x='Probability', y='Precipitation', data=df, color='black')
    plt.ylabel('Precipitation (mm)', fontsize=12)
    plt.xlabel('Probability (%)', fontsize=12)
    plt.title("Probability of Non-Exceedence")
    plt.show()
    
    return p90_value



def max_annual_precipitation(df, name_file, output_dir='Results'):
    """
    Calcula o valor máximo de precipitação anual para cada ano e remove os outliers.
    Em seguida, salva o resultado em um arquivo CSV no diretório especificado.

    Parâmetros:
    - df (DataFrame): DataFrame com colunas 'Year' e 'Precipitation'.
    - name_file (str): Nome base do arquivo de saída.
    - output_dir (str): Diretório onde o arquivo CSV será salvo (padrão: 'Results').

    Retorna:
    - DataFrame com os valores máximos de precipitação anual, excluindo outliers.
    """
    # Remover linhas com valores nulos
    df = df.dropna()
    
    # Agrupar por ano e calcular o valor máximo de precipitação anual
    df_new = df.groupby(['Year'])['Precipitation'].max().reset_index()
    
    # Remover outliers usando a função auxiliar
    df_new = remove_outliers_from_max(df_new)
    
    # Garantir que o diretório de saída exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Caminho completo do arquivo
    output_path = os.path.join(output_dir, f'max_daily_{name_file}.csv')
    
    # Salvar o resultado em um arquivo CSV
    df_new.to_csv(output_path, index=False)
    
    print(f"Arquivo salvo em: {output_path}")
    return df_new



"""
--------------------------------------------------------------------------------------------------------------
----------------------------------------- ANÁLISE DE DADOS SUBDIÁRIOS ----------------------------------------
--------------------------------------------------------------------------------------------------------------
"""


def get_subdaily_extremes(df, interval, dt_min=False, return_max_only=True):
    """
    Calcula os valores máximos e mínimos de precipitação acumulada em intervalos 
    especificados para cada ano presente em um DataFrame. Se return_max_only for True, 
    retorna apenas os máximos.

    Parâmetros:
    df (DataFrame): Um DataFrame que deve conter, pelo menos, uma coluna 'Year' 
                    e dados de precipitação em uma coluna separada.
    interval (int): O intervalo de agregação desejado:
                    - Se 'dt_min' for False, considera 'interval' em horas (para máximos).
                    - Se 'dt_min' for True, considera 'interval' em minutos (para máximos e mínimos).
    dt_min (int, opcional): A resolução temporal dos dados em minutos. 
                            Necessário se 'interval' for em minutos.
    return_max_only (bool, opcional): Se True, retorna apenas os máximos. O padrão é True.

    Retorna:
    DataFrame: Um DataFrame contendo os anos e, dependendo do parâmetro, 
               os máximos e mínimos ou apenas os máximos de precipitação acumulada.
    """
    
    # Obtém a lista de anos únicos do DataFrame
    years_list = df['Year'].unique()
    
    # Inicializa listas para armazenar os máximos e mínimos subdiários
    max_subdaily_list = []
    min_subdaily_list = []

    # Itera sobre cada ano para calcular os extremos de precipitação acumulada
    for year in years_list:
        # Filtra os dados para o ano atual
        df_new = df[df['Year'] == year]
        
        # Agrega a precipitação em intervalos subdiários
        if not dt_min:
            subdaily_list = aggregate_precipitation(df_new, interval)
        else:
            subdaily_list = aggregate_precipitation(df_new, interval, dt_min)

        # Adiciona o máximo e mínimo encontrados às respectivas listas
        max_subdaily_list.append(max(subdaily_list))
        min_subdaily_list.append(min(subdaily_list))

    # Cria um DataFrame resultante com os anos
    if return_max_only:
        df_result = pd.DataFrame({
            'Year': years_list,
            f'Max_{interval}{"h" if dt_min is None else "min"}': max_subdaily_list  # Apenas máximos
        })
    else:
        df_result = pd.DataFrame({
            'Year': years_list,
            f'Max_{interval}{"h" if dt_min is None else "min"}': max_subdaily_list,  # Máximos
            f'Min_{interval}{"h" if dt_min is None else "min"}': min_subdaily_list   # Mínimos
        })

    return df_result



def get_max_subdaily_table(name_file, directory='Results', dt_min=False):
    """
    Calcula os máximos de precipitação acumulada em intervalos subdiários 
    e salva os resultados em um arquivo CSV. O cálculo pode ser realizado 
    para dados horários ou de minutos, dependendo da presença do parâmetro dt_min.

    Parâmetros:
    name_file (str): Nome do arquivo sem extensão que contém dados de precipitação.
    directory (str): Diretório onde os arquivos estão localizados e onde o resultado será salvo.
    dt_min (int, opcional): A resolução temporal dos dados em minutos. Necessário se os dados forem em minutos.

    Retorna:
    None: Salva um arquivo CSV contendo os máximos acumulados por intervalo.
    """
    print('Getting maximum subdaily...')
    
    # Lê o arquivo CSV contendo dados
    if not dt_min:
        df = pd.read_csv(f'{directory}/{name_file}_hourly.csv')
        # Lista dos intervalos em horas
        intervals = [1, 3, 6, 8, 10, 12, 24]
    else:
        df = pd.read_csv(f'{directory}/{name_file}_min.csv')
        # Lista dos intervalos em minutos
        intervals = [5, 10, 15, 20, 25, 30]

    # Cria um DataFrame inicial para armazenar os resultados
    df_final = pd.DataFrame({'Year': df['Year'].unique()})

    # Calcula e mescla os máximos para cada intervalo
    for interval in intervals:
        if not dt_min:
            max_subdaily = get_subdaily_extremes(df, interval)
            print(f'{interval}h done!')
        else:
            max_subdaily = get_subdaily_extremes(df, interval, dt_min)
            print(f'{interval}min done!')
        
        # Mescla os resultados no DataFrame final
        df_final = df_final.merge(max_subdaily, on='Year', how='inner')

    # Exibe o DataFrame final
    print('\n', df_final, '\n')

    # Salva o DataFrame final em um arquivo CSV
    if not dt_min:
        df_final.to_csv(f'{directory}/max_subdaily_{name_file}.csv', index=False)
    else:
        df_final.to_csv(f'{directory}/max_subdaily_min_{name_file}.csv', index=False)

    print('Done!')
    
    return df_final
    
    

def generate_complete_subdaily_table(name_file, directory='Results'):
    """
    Executa o pipeline completo para gerar e mesclar os máximos de precipitação acumulada 
    em intervalos subdiários (minutos e horas), salvando o resultado final em um CSV.

    Parâmetros:
    ----------
    name_file : str
        Nome base do arquivo (sem extensão).
    directory : str
        Diretório onde os arquivos de entrada estão e onde o resultado será salvo.

    Retorna:
    -------
    DataFrame:
        DataFrame final com os máximos acumulados por intervalo em minutos e horas.
    """
    
    print('Iniciando geração da tabela completa de extremos subdiários...\n')

    # Geração para dados em minutos
    df_min = get_max_subdaily_table(name_file, directory=directory, dt_min=True)

    # Geração para dados horários
    df_hour = get_max_subdaily_table(name_file, directory=directory, dt_min=False)

    # Mesclagem dos dois resultados
    df_complete = df_min.merge(df_hour, on='Year', how='inner')

    # Salvando resultado final
    output_path = f'{directory}/max_subdaily_complete_{name_file}.csv'
    df_complete.to_csv(output_path, index=False)

    print('\nPipeline finalizado com sucesso! Arquivo salvo em:', output_path)

    return df_complete