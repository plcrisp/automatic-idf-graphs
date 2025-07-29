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



def distribution_plot_df(df, show_max=False, ax=None, display=True):
    """
    Gera um gráfico de densidade dos dados de precipitação 
    a partir de um DataFrame, com a opção de exibir o maior valor no gráfico.

    Parâmetros:
    - df (DataFrame): Um DataFrame contendo a coluna 'Precipitation'.
    - show_max (bool): Se True, exibe o maior valor de precipitação no gráfico.
    - ax (matplotlib.axes.Axes): Eixo para desenhar o gráfico. Se None, cria um novo.
    - display (bool): Se True, exibe o gráfico. Se False, apenas monta (para uso em subplot).

    Retorna:
    - ax (matplotlib.axes.Axes): O eixo com o gráfico plotado.
    """

    df = df.dropna(subset=['Precipitation'])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    sns.kdeplot(df['Precipitation'], color='skyblue', fill=True, ax=ax)

    if show_max:
        max_value = df['Precipitation'].max()
        ax.annotate(f'Máximo: {max_value} mm',
                    xy=(0.67, 0.9), 
                    xycoords='axes fraction',
                    fontsize=9, color='grey', weight='bold',
                    bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=1'))

    ax.set_title('Distribuição de Precipitação')
    ax.set_xlabel('Precipitação (mm)')
    ax.set_ylabel('Densidade')

    if display:
        plt.show()

    return ax
    


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

    
