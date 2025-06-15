import os
import seaborn as sns
import matplotlib.pyplot as plt

from ...data.processing import remove_outliers_from_max


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