# Importações dos módulos utilitários
from .data.api.inmet import get_inmet_data, load_inmet_station_parameters
from .data.api.cemaden import get_cemaden_data, finalizar_requisicao_por_id
from .data.processing import aggregate_to_csv
from .data.reader import process_data, DataSource
from .analysis.projection.eqm import eqm_downscaling
from .analysis.historical.idf import get_final_idf
from idf_analysis.analysis.historical.intervals import DisaggregationScenario, get_subdaily_from_disaggregation_factors
from idf_analysis.analysis.historical.subdaily import get_max_subdaily_table
from idf_analysis.data.processing import read_csv, verification, fill_missing_data
from idf_analysis.analysis.historical.validation import max_annual_precipitation
from .data.api.climbra import get_climbra_data
from .analysis.projection.dbc import dbc_percentilico




from typing import Optional, List
cemaden_df = read_csv(path='results/cemaden_ac_santana_sao/cemaden_ac_santana_sao_hourly.csv')
aggregate_to_csv(df=cemaden_df, name='cemaden_ac_santana_sao', directory='results/cemaden_ac_santana_sao',include_minutes=True,minute_freq=5)
#inmet_santana = process_data(source=DataSource.INMET, data_path='./datasets/INMET_DAILY_SAO_PAULO_MIRANTE/inmet_daily_sao_paulo_mirante.csv')
#aggregate_to_csv(df=inmet_santana, name='inmet_daily_sao_paulo_mirante', directory='results/inmet_daily_sao_paulo_mirante')
#fig, axes = eqm_downscaling(
  #  name_obs='inmet_daily_sao_paulo_mirante/inmet_daily_sao_paulo_mirante',
  #  name_baseline='CABra467/historical/1980-2013',
  #  name_future='CABra467/ssp245/2015-2100',
  #  dir='results',
  #  plot=True,
#)


'''
cemaden_santana = process_data(source=DataSource.CEMADEN, data_path='./datasets/CEMADEN_SP', site_filter='AC Santana', show_station_counts=True, generate_map=True)
aggregate_to_csv(df=cemaden_santana, name='cemaden_santana', directory='results/cemaden_santana')

inmet_santana = process_data(source=DataSource.INMET_DAILY, data_path='./datasets/INMET_santana/sp-1961-2025.csv')
aggregate_to_csv(df=inmet_santana, name='inmet_santana', directory='results/inmet_santana')
'''
#finalizar_requisicao_por_id(53956, {'codestacao': '355030811A', 'id_tipoestacao': 1, 'nome': 'AC Santana'}, 'SÃO PAULO')
#inmet = get_inmet_data()
#finalizar_requisicao_por_id(53955, {'codestacao': '355030811A', 'id_tipoestacao': 1, 'nome': 'AC Santana'}, 'SÃO PAULO')
#finalizar_requisicao_por_id(53794, {'codestacao': '355030811A', 'id_tipoestacao': 1, 'nome': 'AC Santana'}, 'SÃO PAULO')
#cemaden = get_cemaden_data()

#inmet_df = read_csv(path='results/inmet_santana/inmet_santana_daily.csv')


#incomplete_subdaily_inmet = max_annual_precipitation(df=inmet_df, name_file='inmet_santana', output_dir='results/inmet_santana')
        
#eqm_downscaling(name_obs='inmet_santana', name_gcm_baseline='HADGEM_baseline', name_gcm_future='HADGEM_rcp45', dir_obs='results/inmet_santana', dir_gcm='datasets/GCM')

#finalizar_requisicao_por_id(49444, {'codestacao': '120070801A', 'id_tipoestacao': 1, 'nome': 'Cageacre'}, 'XAPURI')
#finalizar_requisicao_por_id(53605, {'codestacao': '350570802A', 'id_tipoestacao': 1, 'nome': 'Parque Imperial'}, 'BARUERI')
# process_precipitation_series(file_names=['results/cemaden_santana/cemaden_santana_daily.csv','results/inmet_santana/inmet_santana_daily.csv'])
 
'''
# Leitura dos dados agregados
cemaden_santana_daily = read_csv(path='results/cemaden_santana/cemaden_santana_daily.csv')

# Verificação de dias faltantes
verification(cemaden_santana_daily)

cemaden_santana_daily = fill_missing_data(
    path_main='results/cemaden_santana/cemaden_santana_daily.csv',
    path_secondary='results/inmet_santana/inmet_santana_daily.csv',
    overwrite=True
)

verification(cemaden_santana_daily)

# Processamento dos dados diários do INMET
inmet_santana = process_data(source=DataSource.INMET_DAILY, data_path='./datasets/INMET_santana/santana-1961-2025.csv')
aggregate_to_csv(df=inmet_santana, name='inmet_santana', directory='results/inmet_santana')

# Leitura dos dados agregados
i_santana_daily = read_csv(path='results/inmet_santana/inmet_santana_daily.csv')

# Visualização da distribuição das chuvas
distribution_plot_df(i_santana_daily, show_max=True)

# Preenchimento e verificação de dados faltantes
i_santana_daily = fill_missing_data(path='results/inmet_santana/inmet_santana_daily.csv')
verification(i_santana_daily)

# Cálculo do percentil 90 de precipitação
calculate_p90(df=i_santana_daily)

# Extração da precipitação máxima anual
max_i_santana = max_annual_precipitation(df=i_santana_daily, name_file='inmet_santana', output_dir='results/inmet_santana')

# Desagregação subdiária para diferentes cenários
for scenario in [DisaggregationScenario.BASE, DisaggregationScenario.UMIDO, DisaggregationScenario.SECO]:
    get_subdaily_from_disagregation_factors(
        df=max_i_santana,
        scenario=scenario,
        var_value=0.2,
        name_file='inmet_santana',
        directory='results/inmet_santana'
    )

# Plotagens dos máximos subdiários absolutos e relativos
plot_subdaily_maximum(name_file='inmet_santana', directory='results/inmet_santana', var_value=0.2)
plot_subdaily_maximum(name_file='inmet_santana', directory='results/inmet_santana', var_value=0.2, relative=True)

# Ajuste e visualização das distribuições estatísticas
get_distribution(name_file='inmet_santana', directory='results/inmet_santana')
get_distribution(name_file='inmet_santana', directory='results/inmet_santana', disag_factor='_p0.2', duration='Max_24h')


# Geração e plotagem das curvas IDF finais

get_final_idf(
    name_file='inmet_santana',
    directory='results/inmet_santana',
    disag_factor='p0.2',
    save_file=True,
    plot=True,
    durations=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    return_periods=[2, 5, 10, 25, 50, 100, 200, 500, 1000],
    save_plot=True,
    plot_directory='results/inmet_santana/graphs/idf',
    generate_tables=True
)





def complete_precipitation_analysis(
    name_file: str,
    data_path: Optional[str] = None,
    source: DataSource = DataSource.INMET_DAILY,
    var_value: float = 0.2,
    durations: Optional[List[int]] = None,
    return_periods: Optional[List[int]] = None,
    base_dir: str = 'results'
):
    if durations is None:
        durations = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100, 200, 500, 1000]
    if data_path is None:
        data_path = f'./datasets/{name_file}/{name_file}-1961-2025.csv'
    
    directory = f'{base_dir}/{name_file}'

    # Processamento dos dados
    df = process_data(source=source, data_path=data_path)
    aggregate_to_csv(df=df, name=name_file, directory=directory)

    # Leitura e visualização
    daily_df = read_csv(path=f'{directory}/{name_file}_daily.csv')
    distribution_plot_df(daily_df, show_max=True)

    # Preenchimento e verificação de dados faltantes
    daily_df = fill_missing_data(path=f'{directory}/{name_file}_daily.csv')
    verification(daily_df)

    # Cálculo do percentil 90 e máxima anual
    calculate_p90(daily_df)
    max_df = max_annual_precipitation(df=daily_df, name_file=name_file, output_dir=directory)

    # Desagregação para todos os cenários
    for scenario in [DisaggregationScenario.BASE, DisaggregationScenario.UMIDO, DisaggregationScenario.SECO]:
        get_subdaily_from_disagregation_factors(
            df=max_df,
            scenario=scenario,
            var_value=var_value,
            name_file=name_file,
            directory=directory
        )

    # Plotagens dos máximos
    plot_subdaily_maximum(name_file=name_file, directory=directory, var_value=var_value)
    plot_subdaily_maximum(name_file=name_file, directory=directory, var_value=var_value, relative=True)

    # Ajuste e visualização de distribuições
    get_distribution(name_file=name_file, directory=directory)
    get_distribution(name_file=name_file, directory=directory, disag_factor=f'_p{var_value}', duration='Max_24h')

    # Geração da curva IDF final
    
get_final_idf(
        name_file=name_file,
        directory=directory,
        disag_factor=f'p{var_value}',
        save_file=True,
        plot=True,
        durations=durations,
        return_periods=return_periods,
        save_plot=True,
    plot_directory=f'{directory}/graphs/idf',
    generate_tables=True
)
 
    
plot_comparative_absolute(entries=[{"name_file": "inmet_santana", "directory": "results/inmet_santana"},
    {"name_file": "cemaden_santana", "directory": "results/cemaden_santana"}],var_value=0.2,intervalos=['Max_24h', 'Max_1h'], output_directory='results/graphs/comparative_graphs')
'''
#result = get_climbra_data()