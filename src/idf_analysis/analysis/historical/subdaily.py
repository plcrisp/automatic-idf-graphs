import pandas as pd
import os

from .intervals import aggregate_precipitation

def get_subdaily_extremes(df, interval, dt_min=False):
    """
    Calcula os valores máximos de precipitação acumulada em intervalos 
    especificados para cada ano presente em um DataFrame.

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

    # Itera sobre cada ano para calcular os extremos de precipitação acumulada
    for year in years_list:
        # Filtra os dados para o ano atual
        df_new = df[df['Year'] == year]
        
        # Agrega a precipitação em intervalos subdiários
        if not dt_min:
            subdaily_list = aggregate_precipitation(df_new, interval)
        else:
            subdaily_list = aggregate_precipitation(df_new, interval, dt_min)

        # Adiciona o máximo encontrado à lista
        max_subdaily_list.append(round(max(subdaily_list), 2))

    # Cria um DataFrame resultante com os anos
    df_result = pd.DataFrame({
        'Year': years_list,
        f'Max_{interval}{"h" if dt_min is False else "min"}': max_subdaily_list
    })

    return df_result



def get_max_subdaily_table(df, name_file='output', dt_min=False, output_dir='Results'):
    """
    Calcula os máximos de precipitação acumulada em intervalos subdiários 
    a partir de um DataFrame, e salva os resultados em um arquivo CSV.

    Parâmetros:
        df (DataFrame): Dados de entrada com colunas 'Year', 'Precipitation', 
                        e colunas de tempo adequadas (Month, Day, Hour, Minute).
        name_file (str): Nome base do arquivo de saída (sem extensão).
        dt_min (int, opcional): Resolução temporal em minutos (ex: 5, 10). 
                                Se False, assume dados horários.
        output_dir (str): Diretório onde o arquivo CSV será salvo.

    Retorna:
        DataFrame: Tabela com máximos acumulados por intervalo.
    """
    

    if not dt_min:
        intervals = [1, 3, 6, 8, 10, 12, 24]
    else:
        intervals = [5, 10, 15, 20, 25, 30]

    if df.empty:
        print(f"[ERRO] DataFrame vazio fornecido. Encerrando.")
        return

    df_final = pd.DataFrame({'Year': df['Year'].unique()})
    
    for interval in intervals:
        if not dt_min:
            max_subdaily = get_subdaily_extremes(df, interval)
        else:
            max_subdaily = get_subdaily_extremes(df, interval, dt_min=dt_min)

        df_final = df_final.merge(max_subdaily, on='Year', how='inner')

    # Criação de diretório (caso não exista)
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = (
        f'{output_dir}/max_subdaily_{name_file}.csv' if not dt_min
        else f'{output_dir}/max_subdaily_min_{name_file}.csv'
    )

    df_final.to_csv(output_path, index=False)
    print(f"\n[OK] Resultados salvos em: {output_path}")

    return df_final
    
    

def generate_complete_subdaily_table(df, name_file='output', directory='Results'):
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
    df_min = get_max_subdaily_table(df, dt_min=True)

    # Geração para dados horários
    df_hour = get_max_subdaily_table(df, dt_min=False)

    # Mesclagem dos dois resultados
    df_complete = df_min.merge(df_hour, on='Year', how='inner')

    # Salvando resultado final
    output_path = f'{directory}/max_subdaily_complete_{name_file}.csv'
    df_complete.to_csv(output_path, index=False)

    print('\nPipeline finalizado com sucesso! Arquivo salvo em:', output_path)

    return df_complete