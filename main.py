from utils.get_datasets import process_data,DataSource
from utils.data_processing import aggregate_to_csv,read_csv,distribution_plot_df

# CAPTANDO DATASET

inmet_sp = process_data(source=DataSource.INMET_DAILY,data_path='./datasets/INMET_SP/sp-1961-2025.csv')
aggregate_to_csv(df=inmet_sp,name='inmet_sp',directory='results/inmet_sp')

i_sp_yearly = read_csv(name='inmet_sp',var='yearly',directory='results/inmet_sp')
i_sp_monthly = read_csv(name='inmet_sp',var='monthly',directory='results/inmet_sp')
i_sp_daily = read_csv(name='inmet_sp',var='daily',directory='results/inmet_sp')

distribution_plot_df(i_sp_daily)

print(inmet_sp.head(10))