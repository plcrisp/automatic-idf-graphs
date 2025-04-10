"""
Este script carrega dados de precipitação de arquivos CSV, processa máximas subdiárias 
para diferentes períodos e gera gráficos para visualização. Ele calcula estatísticas como 
máximos absolutos, diferenças relativas e distribuições ao longo do tempo, permitindo 
comparações entre diferentes períodos de dados pluviométricos e diferentes referências,
CETESB, CETESB otimizado e o método Bartlett-Lewis.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from collections import Counter



def plot_subdaily_maximum(name_file, directory='Results', var_value=0.2, relative=False):
    """
    Gera gráficos de precipitação subdiária absolutos ou relativos (diferença entre observado e CETESB).

    Parâmetros:
    - name_file (str): Nome base dos arquivos.
    - directory (str): Pasta onde os arquivos estão.
    - var_value (float): Valor percentual para variações da CETESB (ex: 0.2 = 20%).
    - relative (bool): Se True, plota diferenças (observado - CETESB); se False, plota valores absolutos.

    Retorno:
    Nenhum. Gráficos são salvos nas pastas apropriadas.
    """

    print(f"\nIniciando geração de gráficos {'relativos' if relative else 'absolutos'}...\n")

    file_paths = {
        'Observado': f'{directory}/max_subdaily_{name_file}.csv',
        'CETESB': f'{directory}/max_subdaily_{name_file}_ger.csv',
        f'CETESB -{int(var_value * 100)}%': f'{directory}/max_subdaily_{name_file}_m{var_value}.csv',
        f'CETESB +{int(var_value * 100)}%': f'{directory}/max_subdaily_{name_file}_p{var_value}.csv'
    }

    data_frames = []
    for tipo, path in file_paths.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Tipo'] = tipo
            data_frames.append(df)
        else:
            print(f'Arquivo não encontrado: {path} (Ignorando este tipo)')

    if not data_frames:
        print('Nenhum arquivo encontrado. Execução encerrada.')
        return

    df_final = pd.concat(data_frames, ignore_index=True, sort=False)
    intervalos = ['Max_1h', 'Max_6h', 'Max_8h', 'Max_10h', 'Max_12h', 'Max_24h']

    if relative:
        os.makedirs(f'{directory}/graphs/relative', exist_ok=True)

        for intervalo in intervalos:
            if 'Observado' not in df_final['Tipo'].values:
                print(f'Dados observados não encontrados para {intervalo}. Pulando...')
                continue

            df_obs = df_final[df_final['Tipo'] == 'Observado'].reset_index()
            ref_dfs = {}
            for tipo in df_final['Tipo'].unique():
                if tipo == 'Observado':
                    continue
                ref_df = df_final[df_final['Tipo'] == tipo].reset_index()
                if intervalo in df_obs.columns and intervalo in ref_df.columns:
                    ref_df[f'Dif_{intervalo}'] = df_obs[intervalo] - ref_df[intervalo]
                    ref_dfs[tipo] = ref_df

            df_diff = pd.concat(ref_dfs.values(), ignore_index=True, sort=False)

            altura = 5
            largura = max(1.5, min(3.5, len(df_obs['Year'].unique()) * 0.3))

            g = sns.catplot(
                x="Year", y=f'Dif_{intervalo}', hue='Tipo', data=df_diff,
                kind='bar', height=altura, aspect=largura
            )
            g.set_axis_labels('', 'Diferença (mm)')
            g.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.07)
            plt.xticks(rotation=50)
            plt.title(f'Diferença Subdiária {name_file} - {intervalo.replace("Max_", "")}')
            plt.axhline(0, color='black', linestyle='--', linewidth=1)
            plt.savefig(f'{directory}/graphs/relative/{name_file}_{intervalo}_relative.png')
            plt.close()
            print(f'Gráfico de diferença salvo: {name_file}_{intervalo}_relative.png')

    else:
        os.makedirs(f'{directory}/graphs/absolute', exist_ok=True)

        df_final = df_final[['Year'] + intervalos + ['Tipo']]

        for intervalo in intervalos:
            altura = 5
            largura = max(1.5, min(3.5, len(df_final["Year"].unique()) * 0.3))

            g = sns.catplot(
                x="Year", y=intervalo, hue='Tipo', data=df_final,
                kind='bar', height=altura, aspect=largura
            )
            g.set_axis_labels('', 'Precipitação (mm)')
            g.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.07)
            plt.xticks(rotation=50)
            plt.ylim(0, 170)
            plt.title(f'{name_file} - Máximo {intervalo.replace("Max_", "")}')
            plt.savefig(f'{directory}/graphs/absolute/{name_file}_{intervalo}_absolute.png')
            plt.close()
            print(f'Gráfico absoluto salvo: {name_file}_{intervalo}_absolute.png')

    print('\n✅ Finalizado!\n')
    
    

def plot_comparative_absolute(entries, var_value=0.2, intervalos=None, output_directory='graphs/absolute_comparative'):
    """
    Gera gráficos absolutos comparativos de precipitação subdiária entre múltiplos conjuntos de arquivos.

    Parâmetros:
    - entries (list[dict]): Lista com dicionários contendo:
        - 'name_file' (str): Nome base do arquivo.
        - 'directory' (str): Caminho da pasta onde estão os arquivos.
    - var_value (float): Variação percentual da CETESB.
    - intervalos (list[str], opcional): Lista dos intervalos a serem plotados. Usa padrão se None.
    - output_directory (str): Pasta onde os gráficos serão salvos.

    Retorno:
    Nenhum. Gráficos são salvos na pasta especificada.
    """
    import os
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    if intervalos is None:
        intervalos = ['Max_1h', 'Max_6h', 'Max_8h', 'Max_10h', 'Max_12h', 'Max_24h']

    print("\nIniciando geração de gráficos absolutos comparativos...\n")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    tipos_base = {
        'Observado': 'max_subdaily_{}.csv',
        'CETESB': 'max_subdaily_{}_ger.csv',
        f'CETESB -{int(var_value * 100)}%': 'max_subdaily_{}_m{}.csv',
        f'CETESB +{int(var_value * 100)}%': 'max_subdaily_{}_p{}.csv',
    }

    for tipo, file_template in tipos_base.items():
        dataframes = []
        for entry in entries:
            name_file = entry['name_file']
            directory = entry['directory']
            label = entry.get('name_file', name_file)

            if '{}' in file_template:
                path = os.path.join(directory, file_template.format(name_file, var_value))
            else:
                path = os.path.join(directory, file_template.format(name_file))

            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Origem'] = label
                df['Tipo'] = tipo
                dataframes.append(df)
            else:
                print(f'Arquivo não encontrado: {path} (Ignorando para {tipo})')

        if len(dataframes) < 2:
            print(f'Dados insuficientes para o tipo "{tipo}". Pulando...\n')
            continue

        common_years = set(dataframes[0]['Year'])
        for df in dataframes[1:]:
            common_years &= set(df['Year'])

        if not common_years:
            print(f'Nenhum ano em comum para "{tipo}". Pulando...\n')
            continue

        for i in range(len(dataframes)):
            dataframes[i] = dataframes[i][dataframes[i]['Year'].isin(common_years)]

        df_final = pd.concat(dataframes, ignore_index=True, sort=False)

        for intervalo in intervalos:
            if intervalo not in df_final.columns:
                continue

            altura = 5
            largura = max(1.5, min(3.5, len(common_years) * 0.3))

            g = sns.catplot(
                x="Year", y=intervalo, hue="Origem", data=df_final,
                kind='bar', height=altura, aspect=largura
            )
            g.set_axis_labels('', 'Precipitação (mm)')
            g.fig.subplots_adjust(bottom=0.15, top=0.9, left=0.07)
            plt.xticks(rotation=50)
            plt.ylim(0, 170)
            plt.title(f'Comparativo {tipo} - Máximo {intervalo.replace("Max_", "")}')
            save_path = os.path.join(output_directory, f'comparativo_{tipo}_{intervalo}.png')
            plt.savefig(save_path)
            plt.close()
            print(f'Gráfico comparativo salvo: {save_path}')

    print('\n✅ Geração de gráficos comparativos finalizada!\n')




def plot_subdaily_maximum_BL(max_hour):
    """
    Plota comparações de máximas subdiárias de precipitação
    observadas e ajustadas utilizando o método Bartlett-Lewis (BL) e fatores CETESB.

    Args:
        max_hour (int): O número de horas para as quais a máxima subdiária é calculada.
    """

    print('Iniciando o plot das máximas subdiárias relativas BL e observadas\n')

    # Lendo dados de precipitação máxima subdiária
    data_sources = {
        'INMET_aut observed': 'Results/max_subdaily_INMET_aut.csv',
        'INMET_aut CETESB': 'Results/max_subdaily_INMET_aut_ger.csv',
        'MAPLU_usp observed': 'Results/max_subdaily_MAPLU_usp.csv',
        'MAPLU_usp CETESB': 'Results/max_subdaily_MAPLU_usp_ger.csv',
        'INMET_aut BL': 'bartlet_lewis/max_subdaily_INMET_bl.csv',
        'MAPLU_usp BL': 'bartlet_lewis/max_subdaily_MAPLU_usp_bl.csv'
    }

    # Criar um DataFrame vazio para armazenar os dados lidos
    df_list = []
    
    for source, file_path in data_sources.items():
        df = pd.read_csv(file_path)
        df['Type'] = source  # Adiciona coluna para identificar a origem dos dados
        df_list.append(df)

    # Concatenar todos os DataFrames em um único DataFrame
    df_final = pd.concat(df_list, ignore_index=True, sort=False)

    # Selecionar colunas relevantes
    df_final = df_final[['Year', 'Max_1', 'Max_6', 'Max_8', 'Max_10', 'Max_12', 'Max_24', 'Type']]

    # Gráfico de máximas absolutas
    g = sns.catplot(x="Year", y=f"Max_{max_hour}", hue='Type', data=df_final, kind='bar', height=5, aspect=1.5)
    g.set_axis_labels('', 'Precipitação')
    plt.title(f'Máximas Subdiárias INMET e MAPLU - Teste BL - {max_hour}h')
    plt.ylim(0, 170)
    plt.xticks(rotation=50)
    plt.savefig(f'Graphs/subdaily_bl/BL_max{max_hour}_absolute.png')
    print(f'Gráfico das máximas absolutas Max_{max_hour}h gerado com sucesso!\n')

    # Processando os DataFrames para calcular diferenças
    df_processed = {source: df_final[df_final['Type'] == source].reset_index(drop=True) for source in data_sources.keys()}

    # Calculando as diferenças entre máximas observadas e ajustadas
    for source in ['INMET_aut observed', 'MAPLU_usp observed']:
        for method in ['CETESB', 'BL']:
            key = f"{source.split()[0]} {method}"
            df_processed[key]['Dif_{max_hour}'] = df_processed[source]['Max_{max_hour}'] - df_processed[key]['Max_{max_hour}']

    # Somando os erros
    error_sums = {source: df_processed[source]['Dif_{max_hour}'].sum() for source in df_processed if 'CETESB' in source or 'BL' in source}

    # Imprimindo os erros acumulados
    for key, value in error_sums.items():
        print(f'Soma do erro para {key}: {value}\n')

    # Gráfico de diferenças
    df_graph = pd.concat([df_processed['INMET_aut CETESB'], df_processed['INMET_aut BL'],
                          df_processed['MAPLU_usp CETESB'], df_processed['MAPLU_usp BL']], ignore_index=True, sort=False)

    h = sns.catplot(x="Year", y=f"Dif_{max_hour}", hue='Type', data=df_graph, kind='bar', height=5, aspect=1.5)
    h.set_axis_labels('', 'Precipitação')
    plt.title(f'Diferenças Subdiárias INMET e MAPLU - Teste BL - {max_hour}h')
    plt.ylim(-100, 50)
    plt.xticks(rotation=50)
    plt.savefig(f'Graphs/subdaily_bl/BL_max{max_hour}_relative.png')
    print(f'Gráfico das diferenças Max_{max_hour}h gerado com sucesso!\n')
    
