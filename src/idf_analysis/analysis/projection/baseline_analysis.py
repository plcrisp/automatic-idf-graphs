"""
Análise de Dados Climáticos Corretos por Viés - Baseline

Este script realiza uma análise climática baseada em modelos climáticos globais (GCMs) simulando o período de referência (baseline).
Ele aplica diferentes métodos de correção de viés aos dados simulados e gera estatísticas importantes para estudos de mudança climática.

📌 Objetivos principais:
- Processar dados simulados de precipitação de modelos climáticos (baseline)
- Corrigir os dados com diferentes métodos de correção de viés
- Calcular estatísticas como:
    - Percentil 90 (P90) diário
    - Máxima precipitação diária anual
    - Tendência temporal da precipitação

🌍 Modelos Climáticos Utilizados:
- **HADGEM**: Desenvolvido pelo Met Office Hadley Centre (Reino Unido)
- **MIROC5**: Desenvolvido pela Universidade de Tóquio (Japão)
Esses modelos fazem parte de experimentos climáticos globais, como o CMIP, e simulam o clima da Terra com base em equações físicas.

🕰️ Baseline:
O baseline é o período de referência histórico (ex: 1980–2005) simulado pelos modelos climáticos. Ele é usado para validar os modelos contra dados observados e como base de comparação para avaliar mudanças futuras no clima.

🛠️ Métodos de Correção de Viés Aplicados:
- **MD** (*Mean Distribution*): Corrige a média e a distribuição dos dados.
- **PT** (*Power Transformation*): Aplica transformações matemáticas para reduzir o viés.
- **QM** (*Quantile Mapping*): Corrige os quantis da distribuição, muito usado.
- **DBC** (*Double Bias Correction*): Dupla correção que ajusta média e variabilidade.

⚙️ Operações realizadas:
1. Leitura dos arquivos simulados corrigidos por viés (um por modelo e método).
2. Cálculo do P90 para cada série temporal corrigida.
3. Agregação por ano e exportação dos dados anuais.
4. Cálculo da maior precipitação diária em cada ano.
5. Análise de tendência da precipitação total anual e da precipitação máxima diária anual, com base em regressão.

📁 Estrutura de diretórios esperada:
Os arquivos CSV devem estar organizados em:  
`GCM_data/bias_correction/{modelo}_baseline_{método}_daily.csv`

Exemplo de nome de arquivo: `HADGEM_baseline_QM_daily.csv`
"""

from ..historical.extremes import calculate_p90,max_annual_precipitation
from ...data.processing import aggregate_to_csv
from ..historical.trend import get_trend

import pandas as pd


"""
--------------------------------------------------------------------------------------------------------------
-------------------------------------------- ANALISANDO DADOS SIMULADOS --------------------------------------
--------------------------------------------------------------------------------------------------------------
"""


def analyze_baseline_bias_corrected_gcms(
    models: list[str],
    bias_methods: list[str],
    base_path: str = 'GCM_data/bias_correction',
    frequency: str = 'daily',
    group_name: str = 'GCM_baseline',
    alpha: float = 0.05,
    save_csv: bool = True,
):
    """
    Realiza análise de dados simulados no período de baseline com diferentes modelos climáticos e métodos de correção de viés.

    Esta função processa séries temporais diárias simuladas por modelos climáticos globais (GCMs) corrigidos por diversos métodos,
    referentes ao período histórico (baseline). Para cada combinação de modelo e método, calcula-se:

    - Percentil 90 diário (P90)
    - Agregação anual da série
    - Máxima precipitação diária por ano
    - Análise de tendência da precipitação anual e da precipitação máxima diária anual

    Os dados processados podem ser exportados em formato `.csv`, e os resultados estatísticos são exibidos no console.

    Parâmetros:
    ----------
    models : list of str
        Lista dos nomes dos modelos climáticos utilizados (ex: ['HADGEM', 'MIROC5']).

    bias_methods : list of str
        Lista dos métodos de correção de viés aplicados aos dados simulados (ex: ['MD', 'PT', 'QM', 'DBC']).

    base_path : str, default='GCM_data/bias_correction'
        Caminho para o diretório onde estão armazenados os arquivos `.csv` com os dados corrigidos.
        
    frequency : str, default='daily'
        Frequência dos dados a serem analisados.

    group_name : str, default='GCM_baseline'
        Nome do grupo a ser utilizado nas análises de tendência, útil para agrupamentos ou legendas.

    alpha : float, default=0.05
        Nível de significância para o teste estatístico de tendência (ex: 0.05 para 95% de confiança).

    save_csv : bool, default=True
        Se True, os dados agregados por ano e os máximos anuais são salvos como arquivos `.csv`.

    Retorno:
    -------
    None
        Os resultados são impressos no console e, se desejado, arquivos são salvos no diretório especificado.
    """
    
    print('--- Baseline Analysis ---')
    sites_list = []

    for model in models:
        for method in bias_methods:
            name = f"{model}_baseline_{method}"
            file_path = f"{base_path}/{name}_{frequency}.csv"

            try:
                df = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"[!] Arquivo não encontrado: {file_path}")
                continue

            print(f'--> P90 {name}: {calculate_p90(df=df)}')

            if save_csv:
                aggregate_to_csv(df=df,name=name,directory=base_path)

                max_annual_precipitation(df=df,name_file=name,directory=base_path)

            sites_list.append(name)

    print('\n--> Trend analysis')
    print('- Annual precipitation')
    get_trend(var='Year', sites_list=sites_list, group=group_name, alpha=alpha,data_type='mod')

    print('\n- Max_daily')
    get_trend(var='Max_daily', sites_list=sites_list, group=group_name, alpha=alpha, data_type='mod')

    print('\nDone!')