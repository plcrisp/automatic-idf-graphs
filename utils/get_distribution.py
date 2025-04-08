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



def plot_histogram(data, results, n):
    """
    Plota um histograma dos dados fornecidos e sobrepõe as distribuições ajustadas.

    Esta função exibe um histograma representando os dados fornecidos e traça as primeiras
    'n' distribuições ajustadas a partir do ranking fornecido em 'results'. O número de bins
    é calculado usando a função `calculate_bins`. A seleção das distribuições é feita usando
    a função `get_top_fitted_distributions`.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos para o histograma.
        results (dict): Dicionário contendo distribuições ajustadas. 
            A estrutura esperada é {distribuição: (SSE, arg, loc, scale)}.
        n (int): Número de distribuições do ranking a serem sobrepostas no gráfico.

    Retorna:
        None: A função exibe o gráfico, mas não retorna valores.
    """
    if n <= 0:
        raise ValueError("O número de distribuições (n) deve ser maior que zero.")
    if len(results) < n:
        raise ValueError(f"O número de distribuições disponíveis ({len(results)}) é menor que n ({n}).")
    
    # Usa a função `get_top_fitted_distributions` para obter as 'n' melhores distribuições ajustadas
    top_distributions_df = get_top_fitted_distributions(data, results, n)
    
    # Calcula o número ideal de bins usando a função `calculate_bins`
    num_bins = calculate_bins(data)
    
    # Configurações do histograma
    plt.figure(figsize=(10, 5))
    plt.hist(data, density=True, bins=num_bins, ec='white', 
             color=(63/235, 149/235, 170/235), alpha=0.75, label='Dados observados')
    plt.title(f'Histograma e Distribuições Ajustadas\nMáximos de Chuva [mm]', fontsize=14)
    plt.xlabel('Precipitação Máxima [mm]', fontsize=12)
    plt.ylabel('Densidade de Probabilidade', fontsize=12)

    # Define uma paleta de cores distinta
    colors = sns.color_palette("husl", n_colors=n)  # Paleta "husl" do Seaborn

    # Itera sobre as distribuições ajustadas e plota suas PDFs
    for idx, (_, row) in enumerate(top_distributions_df.iterrows()):
        distribution_name = row['distribution']
        sse = row['sse']
        c = row['c']
        loc = row['loc']
        scale = row['scale']
        
        # Recupera a distribuição correspondente ao nome
        distribution = next(dist for dist in get_common_distributions() if dist.name.capitalize() == distribution_name)
        
        # Define os argumentos adicionais (`arg`) se existirem
        arg = (c,) if not np.isnan(c) else ()
        
        # Gera a PDF da distribuição
        x_plot = np.linspace(min(data), max(data), 1000)
        y_plot = distribution.pdf(x_plot, loc=loc, scale=scale, *arg)
        
        # Plota a distribuição com uma cor distinta
        plt.plot(
            x_plot, y_plot, 
            label=f"{distribution_name}: SSE = {sse:.4f}",
            color=colors[idx]
        )

    # Configuração da legenda
    plt.legend(title='Distribuições', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

 

def fit_data(data):
    """
    Ajusta distribuições teóricas comuns aos dados fornecidos, calcula o erro de ajuste (SSE) 
    e retorna os resultados ordenados pelo erro.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos a serem ajustados.

    Retorna:
        dict: Dicionário contendo as distribuições ajustadas com os erros (SSE), 
              parâmetros de ajuste (loc, scale, arg).
    """
    # Validação do parâmetro 'data'
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("O parâmetro 'data' deve ser uma lista ou um array NumPy.")
    if len(data) < 2:
        raise ValueError("O conjunto de dados 'data' deve ter pelo menos dois elementos.")

    # Obtém a lista de distribuições comuns para ajuste
    COMMON_DISTRIBUTIONS = get_common_distributions()

    # Calcula o número ideal de bins para o histograma usando a fórmula de Doane
    # Isso garante que o histograma represente bem a distribuição dos dados.
    num_bins = calculate_bins(data)

    # Gera o histograma dos dados observados
    # `frequencies`: Densidades normalizadas do histograma (soma total = 1).
    # `bin_edges`: Limites dos bins (intervalos) do histograma.
    frequencies, bin_edges = np.histogram(data, bins=num_bins, density=True)

    # Calcula os valores centrais de cada bin
    # Os valores centrais são usados para avaliar as funções de densidade de probabilidade (PDFs)
    # das distribuições teóricas nos mesmos pontos do histograma.
    central_values = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges) - 1)]

    # Inicializa o dicionário para armazenar os resultados de cada distribuição
    results = {}

    # Itera sobre as distribuições comuns para ajustá-las aos dados
    for distribution in COMMON_DISTRIBUTIONS:
        try:
            # Ajusta os parâmetros da distribuição aos dados
            # `params` contém todos os parâmetros ajustados (loc, scale e argumentos adicionais, se houver).
            params = distribution.fit(data)

            # Separa os parâmetros ajustados:
            # - `arg`: Argumentos adicionais da distribuição (ex.: forma, etc.).
            # - `loc`: Parâmetro de localização (deslocamento horizontal).
            # - `scale`: Parâmetro de escala (dispersão).
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Avalia a função de densidade de probabilidade (PDF) da distribuição ajustada
            # nos valores centrais dos bins do histograma.
            pdf_values = distribution.pdf(central_values, loc=loc, scale=scale, *arg)

            # Calcula o erro quadrático somado (SSE) entre o histograma dos dados observados
            # e a PDF ajustada. O SSE mede quão bem a distribuição se ajusta aos dados.
            sse = np.sum(np.power(frequencies - pdf_values, 2.0))

            # Armazena os resultados da distribuição:
            # - `sse`: Erro quadrático somado.
            # - `arg`: Argumentos adicionais da distribuição.
            # - `loc`: Parâmetro de localização.
            # - `scale`: Parâmetro de escala.
            results[distribution] = [sse, arg, loc, scale]
        except Exception as e:
            # Ignora distribuições que falham ao ajustar os dados
            continue

    # Ordena os resultados pelo erro (SSE) em ordem crescente
    # Isso permite identificar as distribuições que melhor se ajustam aos dados.
    sorted_results = {k: results[k] for k in sorted(results, key=lambda x: results[x][0])}

    return sorted_results



def plot_cdf_comparison(data, results, n):
    """
    Avalia o ajuste de distribuições aos dados e plota as funções de distribuição cumulativa (CDF).

    Parâmetros:
        data (array-like): Conjunto de dados numéricos.
        results (dict): Dicionário contendo distribuições ajustadas. 
            Estrutura esperada: {distribuição: (SSE, arg, loc, scale)}.
        n (int): Número de distribuições do ranking a serem avaliadas.
        mean (float, optional): Média teórica ou esperada para testes de ajuste. Default é None.

    Retorna:
        None: A função realiza o gráfico opcional e não retorna valores.
    """
    if n <= 0:
        raise ValueError("O número de distribuições (n) deve ser maior que zero.")
    if len(results) < n:
        raise ValueError(f"O número de distribuições disponíveis ({len(results)}) é menor que n ({n}).")
    
    # Seleciona as primeiras 'n' distribuições do ranking
    selected_distributions = dict(list(results.items())[:n])
    
    # Plota as CDFs das distribuições
    plt.figure(figsize=(10, 5))
    x_plot = np.linspace(min(data), max(data), 1000)  # Calcula uma única vez para todas as distribuições
    
    # Define uma paleta de cores distinta
    colors = sns.color_palette("husl", n_colors=n)  # Paleta "husl" do Seaborn

    for idx, (distribution, (sse, arg, loc, scale)) in enumerate(selected_distributions.items()):
        # Calcula a CDF para a distribuição
        y_plot = distribution.cdf(x_plot, loc=loc, scale=scale, *arg)
        
        # Plota a CDF com uma cor distinta
        plt.plot(
            x_plot, y_plot, 
            label=f"{distribution.name.capitalize()}: SSE = {sse:.4f}",
            color=colors[idx]
        )
    
    # Configurações do gráfico
    plt.title(f'Funções de Distribuição Cumulativa (CDF)\nMáximos de Chuva [mm]', fontsize=14)
    plt.xlabel('Precipitação Máxima [mm]', fontsize=12)
    plt.ylabel('Probabilidade Acumulada', fontsize=12)
    plt.legend(title='Distribuições Ajustadas', bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.show()




def get_top_fitted_distributions(data, results, n):
    """
    Extrai os parâmetros das 'n' melhores distribuições ajustadas aos dados e retorna um DataFrame.

    Parâmetros:
        data (array-like): Conjunto de dados numéricos.
        results (dict): Dicionário contendo distribuições ajustadas.
        n (int): Número de distribuições mais ajustadas a serem retornadas.

    Retorna:
        pd.DataFrame: Um DataFrame com as distribuições, seus erros (SSE) e parâmetros ajustados.
    """
    # Validação dos parâmetros
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("O parâmetro 'data' deve ser uma lista ou um array NumPy.")
    if len(data) < 2:
        raise ValueError("O conjunto de dados 'data' deve ter pelo menos dois elementos.")
    if not isinstance(results, dict):
        raise TypeError("O parâmetro 'results' deve ser um dicionário.")
    if not all(isinstance(k, st.rv_continuous) for k in results.keys()):
        raise TypeError("As chaves do dicionário 'results' devem ser distribuições contínuas do SciPy.")
    if not isinstance(n, int) or n <= 0:
        raise ValueError("O parâmetro 'n' deve ser um número inteiro positivo.")
    if n > len(results):
        raise ValueError(f"O número de distribuições solicitadas ({n}) excede o número disponível ({len(results)}).")

    # Ordena os resultados pelo erro SSE (menor erro primeiro)
    sorted_results = {k: results[k] for k in sorted(results, key=lambda x: results[x][0])}

    # Seleciona as 'n' melhores distribuições
    top_distributions = dict(list(sorted_results.items())[:n])

     # Mapeamento de distribuições para nomes legíveis
    dist_names = {
        dist: dist.name.capitalize() for dist in get_common_distributions()
    }

    # Lista para armazenar os resultados
    result_rows = []

    # Itera sobre as distribuições mais ajustadas
    for distribution, result in top_distributions.items():
        # Nome legível da distribuição
        dist_name = dist_names.get(distribution, distribution.name)

        # Extrai os parâmetros
        sse, arg, loc, scale = result

        # Se 'arg' não for vazio, pega o primeiro valor; caso contrário, define como NaN
        c = arg[0] if len(arg) > 0 else float('nan')

        # Adiciona os resultados à lista
        result_rows.append({
            'distribution': dist_name,
            'sse': sse,
            'c': c,
            'loc': loc,
            'scale': scale
        })

    # Cria o DataFrame com os resultados
    df_result = pd.DataFrame(result_rows)

    return df_result



def get_distribution(name_file, n_distributions=3, duration=None, disag_factor=None, directory='../results'):
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
        n_distributions (int): Número de distribuições mais ajustadas a serem analisadas e exibidas. Padrão é 3.

    Retorna:
        None: Salva os resultados em um arquivo CSV e exibe mensagens de conclusão.
    """
    # Validação do parâmetro n_distributions
    if not isinstance(n_distributions, int) or n_distributions <= 0:
        raise ValueError("O parâmetro 'n_distributions' deve ser um número inteiro positivo.")

    # Constrói o caminho do arquivo de entrada
    if duration is None:
        file_path = f'{directory}/max_daily_{name_file}.csv'
        column_name = 'Precipitation'  # Coluna esperada para dados diários
    else:
        if disag_factor is None:
            raise ValueError("O parâmetro 'disag_factor' deve ser fornecido quando 'duration' é especificado.")
        file_path = f'{directory}/max_subdaily_{name_file}{disag_factor}.csv'
        column_name = duration  # Coluna esperada para dados subdiários

    # Tentativa de leitura do arquivo de dados
    try:
        data_df_original = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        return

    # Verifica se a coluna esperada existe no DataFrame
    if column_name not in data_df_original.columns:
        print(f"Erro: A coluna '{column_name}' não foi encontrada no arquivo.")
        return

    # Filtra a coluna de precipitação e calcula a média
    data_df = data_df_original[[column_name]]
    data = data_df.values.ravel()  # Converte os dados para um array numpy

    # Ajuste de distribuições aos dados (usa distribuições comuns pré-definidas)
    results = fit_data(data)

    # Geração de gráficos
    plot_histogram(data, results, n_distributions)  # Plota o histograma e as distribuições ajustadas
    plot_cdf_comparison(data, results, n_distributions)  # Realiza o teste de bondade de ajuste

    # Obtém os parâmetros das distribuições ajustadas e cria um DataFrame
    df_parameters = get_top_fitted_distributions(data, results, n_distributions)

    # Salva os parâmetros em um arquivo CSV
    output_file = f'{directory}/{name_file}_dist_params.csv'
    df_parameters.to_csv(output_file, index=False)

    print(f"Processamento concluído. Parâmetros das {n_distributions} melhores distribuições salvos em '{output_file}'.")


if __name__ == "__main__":
        
    get_distribution('inmet_conv')
    #get_distribution(name_file='inmet',duration='Max_6h',disag_factor='_p0.2')
    
    
    
