import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pymannkendall as mk



def get_trend(var, sites_list, group, alpha_value=0.05, data_type='obs', plot_graphs=True):
    """
    Realiza a análise de tendência em dados de precipitação usando diferentes variações do 
    teste de Mann-Kendall. Os resultados são armazenados em um CSV para cada grupo e tipo de dado,
    e gráficos de tendência são gerados se plot_graphs=True.

    Parâmetros:
    -----------
    var : str
        Indica a variável usada na análise ('Year' para dados anuais ou 'Max_daily' para dados diários).
    sites_list : list
        Lista de nomes dos sites para os quais a análise será feita.
    alpha_value : float
        Valor de significância para os testes (ex.: 0.05).
    group : str
        Nome do grupo ao qual os sites pertencem (usado para nomear arquivos de saída).
    data_type : str, opcional
        Tipo de dado: 'obs' para dados observados, 'mod' para dados modelados por GCM.
    plot_graphs : bool, opcional
        Indica se gráficos de tendência devem ser plotados.

    Retorno:
    --------
    Salva um arquivo CSV contendo os resultados da análise de tendência para cada site e teste,
    além de gráficos de tendência, se plot_graphs=True.
    """
    print('Running get_trend...')
    
    # Dicionário para armazenar os resultados de todos os testes
    trend_results = {
        'Site': [], 'Test_Type': [], 'Tau': [], 'p_value': [], 'Trend': [], 'h': [],
        'z': [], 's': [], 'var_s': [], 'Slope': [], 'Intercept': []
    }
    
    def store_results(site, test_type, result):
        """
        Armazena os resultados de um teste específico no dicionário principal.
        
        Parâmetros:
        -----------
        site : str
            Nome do site para o qual o teste foi realizado.
        test_type : str
            Tipo do teste de Mann-Kendall (Original, Hamed-Rao, etc.).
        result : dict
            Dicionário com os resultados do teste.
        """
        trend_results['Site'].append(site)
        trend_results['Test_Type'].append(test_type)
        trend_results['Tau'].append(result['Tau'])
        trend_results['p_value'].append(result['p'])
        trend_results['Trend'].append(result['trend'])
        trend_results['h'].append(result['h'])
        trend_results['z'].append(result['z'])
        trend_results['s'].append(result['s'])
        trend_results['var_s'].append(result['var_s'])
        trend_results['Slope'].append(result['slope'])
        trend_results['Intercept'].append(result['intercept'])

    from scipy.stats import linregress

    def plot_trend_graph(df, site, test_name, slope, intercept, var):
        plt.figure(figsize=(10, 6))
        x_var = 'Year'
        y_var = 'Precipitation'

        # Plota os pontos de dados
        sns.scatterplot(data=df, x=x_var, y=y_var, color='blue', label='Observações')
        
        # Calcula a linha de regressão linear
        regression = linregress(df[x_var], df[y_var])
        y_values = regression.slope * df[x_var] + regression.intercept
        
        plt.plot(df[x_var], y_values, color='green', label='Linha de Regressão Linear')
        
        # Calcula a linha de tendência (Sen's Slope)
        trend_values = slope * df[x_var] + intercept
        plt.plot(df[x_var], trend_values, color='red', label=f'Linha de Tendência ({test_name})')
        
        # Configurações do gráfico
        plt.title(f'Tendência de Precipitação - {site} ({test_name})', fontsize=14)
        plt.xlabel(x_var, fontsize=12)
        plt.ylabel('Precipitação (mm)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f'Graphs/{site}_{test_name}_{var}_trend_plot.png')
        plt.close()


    # Mapeamento dos testes de Mann-Kendall disponíveis
    test_functions = {
        'Original': mk.original_test,
        'Hamed-Rao': mk.hamed_rao_modification_test,
        'Yue-Wang': mk.yue_wang_modification_test,
        'Trend-Free': mk.trend_free_pre_whitening_modification_test,
        'Pre-Whitening': mk.pre_whitening_modification_test
    }

    # Itera sobre todos os sites fornecidos
    for site in sites_list:
        # Define o caminho do arquivo CSV com base no tipo de dado e variável
        if data_type == 'obs':
            file_path = f'Results/{site}_yearly.csv' if var == 'Year' else f'Results/max_daily_{site}.csv'
        elif data_type == 'mod':
            file_path = f'GCM_data/bias_correction/{site}_yearly.csv' if var == 'Year' else f'GCM_data/bias_correction/max_daily_{site}.csv'
        else:
            raise ValueError("Invalid data type. Use 'obs' or 'mod'.")
        
        # Lê o arquivo CSV e remove valores nulos
        df = pd.read_csv(file_path).dropna(subset=['Precipitation'])
        print(f'--- {group} / {site} ---\n')

        try:
            # Aplica todos os testes de Mann-Kendall
            for test_name, test_func in test_functions.items():
                result = test_func(df[['Precipitation']], alpha=alpha_value)
                store_results(site, test_name, {
                    'Tau': result[4], 'p': result[2], 'trend': result[0], 'h': result[1],
                    'z': result[3], 's': result[5], 'var_s': result[6],
                    'slope': result[7], 'intercept': result[8]
                })
                
                # Gera gráficos se plot_graphs=True
                if plot_graphs and result[7] is not None and result[8] is not None:
                    plot_trend_graph(df, site, test_name, result[7], result[8], var)
            print('')
        except ZeroDivisionError:
            print('Division by zero!! - Not possible to perform trend analysis\n')
        except Exception as e:
            print(f'ALERT!!! Something else went wrong: {e}')
            raise

    # Cria um DataFrame com os resultados de todos os testes
    df_trend_result = pd.DataFrame(trend_results)
    
    # Define o caminho do arquivo de saída com base no tipo de dado
    output_path = (
        f'Results/{group}_{var}_trend_result.csv'
        if data_type == 'obs'
        else f'GCM_data/bias_correction/{group}_{var}_trend_result.csv'
    )
    
    # Salva o DataFrame em um arquivo CSV
    df_trend_result.to_csv(output_path, index=False, encoding='latin1')
    
