"""
Este script fornece um conjunto abrangente de funções para analisar dados de 
precipitação (diários ou subdiários), ajustando distribuições estatísticas, 
gerando visualizações e salvando os parâmetros das melhores distribuições ajustadas.

Principais Funcionalidades:
- Fluxo de Trabalho Unificado: Combina a análise de dados diários e subdiários de precipitação.
- Ajuste de Distribuições: Ajusta automaticamente distribuições estatísticas comuns aos dados 
  e avalia o ajuste usando o erro quadrático somado (SSE - Sum of Squared Errors).
- Ferramentas de Visualização: Gera histogramas e funções de distribuição cumulativa (CDF) 
  com legendas claras e paletas de cores distintas para melhor interpretabilidade.
- Personalização: Permite ao usuário especificar o número de distribuições a serem analisadas 
  e visualizadas, além do tipo de dado (diário ou subdiário).
- Saída: Salva os parâmetros das melhores distribuições ajustadas em um arquivo CSV para uso 
  posterior ou relatórios.

Funções Principais:
1. `get_distribution`:
   - Ponto principal de entrada para análise de dados de precipitação.
   - Lida com dados diários ou subdiários, ajusta distribuições, gera gráficos e salva resultados.
   - Parâmetros:
     - `name_file`: Nome base do arquivo de entrada (sem extensão).
     - `duration`: Especifica a coluna de duração para dados subdiários (ex.: 'Max_1h', 'Max_6h').
     - `disag_factor`: Fator usado no nome dos arquivos subdiários (ex.: '0.2', '0.3').
     - `directory`: Diretório onde os arquivos de entrada/saída estão localizados (padrão: 'results').
     - `n_distributions`: Número de melhores distribuições a serem analisadas e visualizadas (padrão: 3).

2. `fit_data`:
   - Ajusta distribuições estatísticas comuns aos dados e calcula o SSE para cada ajuste.
   - Retorna um dicionário de resultados ordenado pelo SSE.

3. `plot_histogram`:
   - Plota um histograma dos dados observados e sobrepõe as PDFs das melhores distribuições ajustadas.
   - Inclui títulos descritivos, rótulos dos eixos e legendas adaptadas para dados de precipitação.

4. `plot_cdf_comparison`:
   - Plota as funções de distribuição cumulativa (CDFs) das melhores distribuições ajustadas.
   - Destaca visualmente o ajuste com cores distintas e legendas detalhadas.

5. `get_top_fitted_distributions`:
   - Extrai os parâmetros das melhores distribuições ajustadas e organiza-os em um DataFrame.
   
6. `calculate_bins`:
   - Calcula o número ideal de bins para um histograma usando a fórmula de Doane.
   
7. `get_common_distributions`:
   - Retorna uma lista de distribuições contínuas comuns do SciPy.

Exemplo de Uso:
---------------
Para analisar dados diários de precipitação:
    get_distribution(name_file='inmet_conv', n_distributions=3)

Para analisar dados subdiários de precipitação (ex.: Max_6h):
    get_distribution(
        name_file='inmet_conv',
        duration='Max_6h',
        disag_factor=0.8,
        directory='results',
        n_distributions=5
    )

Dependências:
--------------
- Bibliotecas Python: pandas, numpy, scipy.stats, matplotlib, seaborn

Notas Importantes:
-------------------
- Certifique-se de que os arquivos CSV de entrada sigam a estrutura esperada:
  - Dados diários: Coluna chamada 'Precipitation'.
  - Dados subdiários: Colunas nomeadas de acordo com a `duration` especificada (ex.: 'Max_1h', 'Max_6h').
- Os arquivos de saída (CSV com parâmetros ajustados) são salvos no diretório especificado.
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import math
import seaborn as sns

from enum import Enum



class CommonDistributions(Enum):
    NORMAL = ("Normal", st.norm)
    LOGNORMAL = ("Log-normal", st.lognorm)
    PARETO = ("Pareto", st.pareto)
    GUMBEL_R = ("Gumbel (direita)", st.gumbel_r)
    GEV = ("Generalized Extreme Value (GEV)", st.genextreme)
    GENLOGISTIC = ("Generalized Logistic", st.genlogistic)



def get_common_distributions():
    """
    Retorna uma lista de distribuições contínuas comuns do SciPy.

    Retorna:
        list: Lista de distribuições contínuas.
    """
    return [
        st.norm,          # Normal 
        st.lognorm,       # Log-normal 
        st.pareto,        # Pareto
        st.gumbel_r,      # Gumbel (direita) 
        st.genextreme,    # Generalized Extreme Value (GEV)
        st.genlogistic,   # Generalized Logistic
    ]
    


def calculate_bins(data):
    """
    Calcula o número ideal de bins para um histograma usando a fórmula de Doane.

    A fórmula de Doane é uma modificação da regra de Sturges, ajustando o cálculo para
    levar em conta a assimetria dos dados (skewness). É especialmente útil para conjuntos
    de dados que não seguem uma distribuição normal.

    Referência:
    https://en.wikipedia.org/wiki/Histogram#Doane's_formula

    Parâmetros:
        data (list ou array-like): Conjunto de dados numéricos.

    Retorna:
        int: Número ideal de bins para o histograma.
    """
    # Número de observações
    N = len(data)
    
    if N <= 1:
        raise ValueError("O conjunto de dados deve conter pelo menos 2 elementos.")
    
    # Calcula o coeficiente de assimetria (skewness)
    skewness = st.skew(data)
    
    # Calcula o desvio padrão do coeficiente de assimetria
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    
    # Aplica a fórmula de Doane
    num_bins = 1 + math.log2(N) + math.log2(1 + abs(skewness) / sigma_g1)
    
    # Retorna o número de bins arredondado para o inteiro mais próximo
    return round(num_bins)



def plot_histogram(data, results, n, distributions=None):
    """
    Plota um histograma dos dados fornecidos e sobrepõe as distribuições ajustadas.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos para o histograma.
        results (dict): Dicionário contendo distribuições ajustadas. 
            Estrutura esperada: {CommonDistributions: (SSE, arg, loc, scale)}.
        n (int): Número de distribuições do ranking a serem sobrepostas.
        distributions (list of CommonDistributions, optional): Lista de distribuições específicas. 
            Se fornecida, ignora o 'n' e plota apenas essas.

    Retorna:
        None: Apenas exibe o gráfico.
    """
    if distributions is not None:
        if not all(isinstance(d, CommonDistributions) for d in distributions):
            raise TypeError("O parâmetro 'distributions' deve ser uma lista de CommonDistributions.")
        selected_distributions = {d: results[d] for d in distributions if d in results}
        if not selected_distributions:
            raise ValueError("Nenhuma das distribuições fornecidas está presente nos resultados.")
    else:
        if n <= 0:
            raise ValueError("O número de distribuições (n) deve ser maior que zero.")
        if len(results) < n:
            raise ValueError(f"O número de distribuições disponíveis ({len(results)}) é menor que n ({n}).")
        selected_distributions = dict(list(results.items())[:n])

    num_bins = calculate_bins(data)
    plt.figure(figsize=(10, 5))
    plt.hist(data, density=True, bins=num_bins, ec='white', 
             color=(63/235, 149/235, 170/235), alpha=0.75, label='Dados observados')
    plt.title(f'Histograma e Distribuições Ajustadas\nMáximos de Chuva [mm]', fontsize=14)
    plt.xlabel('Precipitação Máxima [mm]', fontsize=12)
    plt.ylabel('Densidade de Probabilidade', fontsize=12)

    colors = sns.color_palette("husl", n_colors=len(selected_distributions))
    x_plot = np.linspace(min(data), max(data), 1000)

    for idx, (dist_enum, (sse, arg, loc, scale)) in enumerate(selected_distributions.items()):
        y_plot = dist_enum.value[1].pdf(x_plot, loc=loc, scale=scale, *arg)
        plt.plot(
            x_plot, y_plot, 
            label=f"{dist_enum.name.capitalize()}: SSE = {sse:.4f}",
            color=colors[idx]
        )

    plt.legend(title='Distribuições', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

 

def fit_data(data, distributions=None):
    """
    Ajusta distribuições teóricas aos dados fornecidos, calcula o erro de ajuste (SSE) 
    e retorna os resultados ordenados pelo erro.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos a serem ajustados.
        distributions (list of CommonDistributions, opcional): Enum de distribuições a serem usadas.
                                                               Se None, usa as distribuições padrão.

    Retorna:
        dict: Dicionário contendo as distribuições ajustadas com os erros (SSE), 
              parâmetros de ajuste (loc, scale, arg).
    """
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("O parâmetro 'data' deve ser uma lista ou um array NumPy.")
    if len(data) < 2:
        raise ValueError("O conjunto de dados 'data' deve ter pelo menos dois elementos.")

    # Verificação e obtenção das distribuições
    if distributions is None:
        distributions = list(CommonDistributions)
    else:
        if not all(isinstance(d, CommonDistributions) for d in distributions):
            raise TypeError("O parâmetro 'distributions' deve ser uma lista de CommonDistributions.")

    # Histograma
    num_bins = calculate_bins(data)
    frequencies, bin_edges = np.histogram(data, bins=num_bins, density=True)
    central_values = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]

    results = {}

    for enum_dist in distributions:
        distribution = enum_dist.value[1]
        try:
            params = distribution.fit(data)
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]
            pdf_values = distribution.pdf(central_values, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(frequencies - pdf_values, 2.0))
            results[enum_dist] = [sse, arg, loc, scale]
        except Exception:
            continue

    sorted_results = {k: results[k] for k in sorted(results, key=lambda x: results[x][0])}
    return sorted_results



def plot_cdf_comparison(data, results, n, distributions=None):
    """
    Avalia o ajuste de distribuições aos dados e plota as funções de distribuição cumulativa (CDF).

    Parâmetros:
        data (array-like): Conjunto de dados numéricos.
        results (dict): Dicionário contendo distribuições ajustadas. 
            Estrutura esperada: {CommonDistributions: (SSE, arg, loc, scale)}.
        n (int): Número de distribuições do ranking a serem avaliadas.
        distributions (list of CommonDistributions, optional): Lista de distribuições a serem usadas.
            Se None, usa as 'n' melhores do ranking.

    Retorna:
        None: A função apenas exibe o gráfico.
    """
    if distributions is not None:
        if not all(isinstance(d, CommonDistributions) for d in distributions):
            raise TypeError("O parâmetro 'distributions' deve ser uma lista de CommonDistributions.")
        selected_distributions = {d: results[d] for d in distributions if d in results}
        if not selected_distributions:
            raise ValueError("Nenhuma das distribuições fornecidas está presente nos resultados.")
    else:
        if n <= 0:
            raise ValueError("O número de distribuições (n) deve ser maior que zero.")
        if len(results) < n:
            raise ValueError(f"O número de distribuições disponíveis ({len(results)}) é menor que n ({n}).")
        selected_distributions = dict(list(results.items())[:n])
    
    # Plota as CDFs
    plt.figure(figsize=(10, 5))
    x_plot = np.linspace(min(data), max(data), 1000)
    colors = sns.color_palette("husl", n_colors=len(selected_distributions))

    for idx, (dist_enum, (sse, arg, loc, scale)) in enumerate(selected_distributions.items()):
        y_plot = dist_enum.value[1].cdf(x_plot, loc=loc, scale=scale, *arg)
        plt.plot(
            x_plot, y_plot,
            label=f"{dist_enum.name.capitalize()}: SSE = {sse:.4f}",
            color=colors[idx]
        )

    plt.title(f'Funções de Distribuição Cumulativa (CDF)\nMáximos de Chuva [mm]', fontsize=14)
    plt.xlabel('Precipitação Máxima [mm]', fontsize=12)
    plt.ylabel('Probabilidade Acumulada', fontsize=12)
    plt.legend(title='Distribuições Ajustadas', bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()




def get_top_fitted_distributions(data, results, n, distributions=None):
    """
    Extrai os parâmetros das distribuições ajustadas aos dados e retorna um DataFrame.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos.
        results (dict): Dicionário contendo distribuições ajustadas.
            Esperado: {CommonDistributions: (SSE, arg, loc, scale)}
        n (int): Número de distribuições mais ajustadas a serem retornadas (ignorado se `distributions` for fornecido).
        distributions (list of CommonDistributions, optional): Lista de distribuições específicas a incluir.

    Retorna:
        pd.DataFrame: Um DataFrame com os nomes das distribuições, seus SSEs e parâmetros ajustados.
    """
    
    # Validação dos dados
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("O parâmetro 'data' deve ser uma lista ou um array NumPy.")
    if len(data) < 2:
        raise ValueError("O conjunto de dados 'data' deve ter pelo menos dois elementos.")
    if not isinstance(results, dict):
        raise TypeError("O parâmetro 'results' deve ser um dicionário.")
    if not all(isinstance(k, CommonDistributions) for k in results.keys()):
        raise TypeError("As chaves do dicionário 'results' devem ser instâncias de CommonDistributions.")

    if distributions is not None:
        if not all(isinstance(d, CommonDistributions) for d in distributions):
            raise TypeError("O parâmetro 'distributions' deve ser uma lista de CommonDistributions.")
        selected_distributions = {d: results[d] for d in distributions if d in results}
        if not selected_distributions:
            raise ValueError("Nenhuma das distribuições fornecidas está presente nos resultados.")
    else:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("O parâmetro 'n' deve ser um número inteiro positivo.")
        if n > len(results):
            raise ValueError(f"O número de distribuições solicitadas ({n}) excede o número disponível ({len(results)}).")
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1][0]))  # ordena pelo SSE
        selected_distributions = dict(list(sorted_results.items())[:n])

    # Construção do DataFrame
    result_rows = []
    for dist_enum, (sse, arg, loc, scale) in selected_distributions.items():
        c = arg[0] if len(arg) > 0 else float('nan')
        result_rows.append({
            'distribution': dist_enum.name.capitalize(),
            'sse': sse,
            'c': c,
            'loc': loc,
            'scale': scale
        })

    return pd.DataFrame(result_rows)



def get_distribution(name_file, n=3, duration=None, disag_factor=None, directory='../results', distributions=None):
    """
    Função principal para carregar, analisar e ajustar distribuições a dados de precipitação (diários ou subdiários).

    O processo inclui:
    1. Carregar os dados de precipitação.
    2. Ajustar distribuições selecionadas aos dados (usando distribuições comuns pré-definidas).
    3. Gerar histogramas e realizar testes de bondade de ajuste.
    4. Obter parâmetros das distribuições ajustadas e salvar os resultados.

    Parâmetros:
        name_file (str): Nome base do arquivo de dados (sem extensão).
        duration (str, optional): Duração do evento de precipitação (ex.: 'Max_1h', 'Max_24h'). 
                                  Se None, assume-se que os dados são diários.
        disag_factor (float, optional): Fator de desagregação para nomear arquivos subdiários (ex.: '_p0.2', '_m0.3'). 
                                       Ignorado se `duration` for None.
        directory (str): Diretório onde os arquivos estão localizados. Padrão é 'results'.
        n (int): Número de distribuições mais ajustadas a serem analisadas e exibidas. Padrão é 3.
        distributions (list of CommonDistributions, optional): Lista de distribuições específicas a serem utilizadas. 
                                                               Se fornecida, ignora `n_distributions`.

    Retorna:
        None: Salva os resultados em um arquivo CSV e exibe mensagens de conclusão.
    """
    if distributions is None:
        if not isinstance(n, int) or n <= 0:
            raise ValueError("O parâmetro 'n' deve ser um número inteiro positivo.")

    # Constrói o caminho do arquivo de entrada
    if duration is None:
        file_path = f'{directory}/max_daily_{name_file}.csv'
        column_name = 'Precipitation'
    else:
        if disag_factor is None:
            raise ValueError("O parâmetro 'disag_factor' deve ser fornecido quando 'duration' é especificado.")
        file_path = f'{directory}/max_subdaily_{name_file}{disag_factor}.csv'
        column_name = duration

    # Tentativa de leitura
    try:
        data_df_original = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return

    if column_name not in data_df_original.columns:
        print(f"Erro: A coluna '{column_name}' não foi encontrada no arquivo.")
        return

    # Dados
    data_df = data_df_original[[column_name]]
    data = data_df.values.ravel()

    # Ajuste das distribuições
    results = fit_data(data,distributions=distributions)

    # Gráficos
    plot_histogram(data, results, n=n, distributions=distributions)
    plot_cdf_comparison(data, results, n=n, distributions=distributions)

    # Tabela de parâmetros
    df_parameters = get_top_fitted_distributions(data, results, n=n, distributions=distributions)

    # Salvamento
    output_file = f'{directory}/{name_file}_dist_params.csv'
    df_parameters.to_csv(output_file, index=False)

    print(f"Processamento concluído. Parâmetros salvos em '{output_file}'.")


if __name__ == "__main__":
        
    get_distribution(name_file='inmet_santana',directory='results/inmet_santana',distributions=[CommonDistributions.NORMAL,CommonDistributions.GEV,CommonDistributions.GUMBEL_R])
    #get_distribution(name_file='inmet',duration='Max_6h',disag_factor='_p0.2')
    
    
    
