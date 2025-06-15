import pandas as pd
from .intervals import aggregate_precipitation

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