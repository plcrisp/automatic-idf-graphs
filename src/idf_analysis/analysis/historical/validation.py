import os
import seaborn as sns
import matplotlib.pyplot as plt

from ...data.processing import verification, fill_missing_data, read_csv, remove_outliers_from_max
from ...core.correlation import left_join_precipitation


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



def process_precipitation_series(file_names):
    """
    Processa séries temporais de precipitação, realizando:

    1. Leitura e verificação da integridade das séries (presença de dias faltantes).
    2. Preenchimento de lacunas (gaps) com base na seguinte lógica:
       - Se ao menos um dataset estiver completo, ele será usado como referência 
         para preencher os datasets incompletos.
       - Se todos estiverem incompletos, cada um será preenchido individualmente.
    3. União das séries em um único DataFrame, com cálculo da precipitação média diária.
    4. Cálculo das somas acumuladas (precipitação acumulada) para cada estação e para a média.
    5. Geração de gráficos de dupla massa (dispersão entre acumulado individual e acumulado médio).

    Parâmetros:
        file_names (list): Lista com os nomes dos arquivos (sem extensão).

    Retorno:
        None. Exibe informações no console e gera gráficos com os dados processados.
    """

    def load_and_verify(file_name):
        print(f"[INFO] Lendo e verificando: {file_name}")
        df = read_csv(file_name)
        result = verification(df)
        return df, result["status"]

    # ETAPA 1: LEITURA E VERIFICAÇÃO DE GAPS
    verification_results = {}
    dataframes = {}

    for name in file_names:
        df, status = load_and_verify(name)
        dataframes[name] = df
        verification_results[name] = status

    # ETAPA 2: PREENCHIMENTO DE GAPS
    complete_paths = [name for name, status in verification_results.items() if status == 'complete']
    incomplete_paths = [name for name, status in verification_results.items() if status == 'incomplete']

    if complete_paths:
        reference = complete_paths[0]
        print(f"[INFO] Usando '{reference}' como referência para preenchimento.")
        for name in incomplete_paths:
            print(f"[INFO] Preenchendo '{name}' com base em '{reference}'...")
            dataframes[name] = fill_missing_data(path_main=name, path_secondary=reference, overwrite=False)
    elif incomplete_paths:
        print("[INFO] Nenhum dataset completo encontrado. Preenchendo todos individualmente...")
        for name in incomplete_paths:
            print(f"[INFO] Preenchendo '{name}' individualmente...")
            dataframes[name] = fill_missing_data(path_main=name)
    else:
        print("\n[INFO] Todos os datasets estão completos. Nenhum preenchimento necessário.\n")

    # ETAPA 3: UNIÃO E PROCESSAMENTO
    print("[INFO] Unindo séries e calculando média diária...")
    df = left_join_precipitation(*dataframes.values())
    df.columns = ['Date'] + [f'P_{name}' for name in file_names]

    df = df.dropna()
    df['P_average'] = df.iloc[:, 1:].mean(axis=1)

    for col in df.columns[1:]:
        df[f'Pacum_{col}'] = df[col].fillna(0).cumsum()

    # ETAPA 4: PLOTAGEM
    print("[INFO] Gerando gráficos de dupla massa...")
    sns.set_context("talk", font_scale=0.8)
    fig, axes = plt.subplots(1, len(file_names), figsize=(20, 6), sharey=True)

    for ax, name in zip(axes, file_names):
        sns.scatterplot(
            x="Pacum_P_average",
            y=f"Pacum_P_{name}",
            data=df,
            ax=ax
        )
        ax.set_xlabel("Média Pacum (mm)")
        ax.set_ylabel(f"Pacum {name} (mm)")
        ax.set_title(f"Dispersão de {name}")

    plt.tight_layout()
    plt.show()
    print("[INFO] Processamento concluído.")