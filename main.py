# Importações dos módulos utilitários
from utils.get_datasets import process_data, DataSource
from utils.data_processing import aggregate_to_csv, read_csv, distribution_plot_df
from utils.error_correction import verification, fill_missing_data
from utils.extreme_precipitation_analysis import calculate_p90, max_annual_precipitation
from utils.extreme_precipitation_visualization import plot_subdaily_maximum
from utils.intervals_manipulation import get_subdaily_from_disagregation_factors, DisaggregationScenario
from utils.get_distribution import get_distribution
from utils.idf_generator import *

# Processamento dos dados diários do INMET
inmet_sp = process_data(source=DataSource.INMET_DAILY, data_path='./datasets/INMET_SP/sp-1961-2025.csv')
aggregate_to_csv(df=inmet_sp, name='inmet_sp', directory='results/inmet_sp')

# Leitura dos dados agregados
i_sp_yearly = read_csv(path='results/inmet_sp/inmet_sp_yearly.csv')
i_sp_monthly = read_csv(path='results/inmet_sp/inmet_sp_monthly.csv')
i_sp_daily = read_csv(path='results/inmet_sp/inmet_sp_daily.csv')

# Visualização da distribuição das chuvas
distribution_plot_df(i_sp_daily, show_max=True)

# Preenchimento e verificação de dados faltantes
i_sp_daily = fill_missing_data(path='results/inmet_sp/inmet_sp_daily.csv')
verification(i_sp_daily)

# Cálculo do percentil 90 de precipitação
calculate_p90(df=i_sp_daily)

# Extração da precipitação máxima anual
max_i_sp = max_annual_precipitation(df=i_sp_daily, name_file='inmet_sp', output_dir='results/inmet_sp')

# Desagregação subdiária para diferentes cenários
for scenario in [DisaggregationScenario.BASE, DisaggregationScenario.UMIDO, DisaggregationScenario.SECO]:
    get_subdaily_from_disagregation_factors(
        df=max_i_sp,
        scenario=scenario,
        var_value=0.2,
        name_file='inmet_sp',
        directory='results/inmet_sp'
    )

# Plotagens dos máximos subdiários absolutos e relativos
plot_subdaily_maximum(name_file='inmet_sp', directory='results/inmet_sp', var_value=0.2)
plot_subdaily_maximum(name_file='inmet_sp', directory='results/inmet_sp', var_value=0.2, relative=True)

# Ajuste e visualização das distribuições estatísticas
get_distribution(name_file='inmet_sp', directory='results/inmet_sp')
get_distribution(name_file='inmet_sp', directory='results/inmet_sp', disag_factor='_p0.2', duration='Max_24h')

# Geração e plotagem das curvas IDF finais
get_final_idf_params(
    name_file='inmet_sp',
    directory='results/inmet_sp',
    disag_factor='p0.2',
    save_file=True,
    plot=True,
    durations=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    return_periods=[2, 5, 10, 25, 50, 100, 200, 500, 1000],
    save_plot=True,
    plot_directory='results/inmet_sp/graphs/idf',
)
