import pandas as pd
import numpy as np
from datetime import date
from datetime import timedelta


## Bias correction pelo CMhyd
def get_CMhyd_file(name_gcm, scenario, bias_correction_method):
    if name_gcm == 'HADGEM':
        if scenario == 'baseline':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Mapping distribution\PCP\MOD\DistributionMapping\historical\historico_dm_hist.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Power transformation\PCP\MOD\PowerTransformation\historical\historico_pt_hist.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 4.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 4.5 hadgem 5km\saida mapping\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 4.5 hadgem 5km\saida power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 8.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Mapping distribution\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 hadgem 5km\Power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        else:
            print('Scenario not bias corrected through CMhyd')
    
    elif name_gcm == 'MIROC5':
        if scenario == 'baseline':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida mapping\PCP\MOD\DistributionMapping\historical\historico_dm_hist.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida power transformation\PCP\MOD\PowerTransformation\historical\historico_pt_hist.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 4.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 4.5 miroc5\saida mapping\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 4.5 miroc5\saida power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        elif scenario == 'rcp 8.5':
            if bias_correction_method == 'MD':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida mapping\PCP\MOD\DistributionMapping\EXP\\futuro_dm_sce.txt', header = None)
            elif bias_correction_method == 'PT':
                df = pd.read_csv('CMhyd\projeta 8.5 miroc5\saida power transformation\PCP\MOD\PowerTransformation\EXP\\futuro_pt_sce.txt', header = None)
            else:
                print('Bias correction not performed')
        else:
            print('Scenario not bias corrected through CMhyd')        
                        
            

    list_complete = df[0].to_list()
    prec_list = []
    for i in range(1,len(list_complete)):
        if list_complete[i] == -99.0:
            prec = 0
        else:
            prec = list_complete[i]
        prec_list.append(prec)
    #print(prec_list)

    date_begin_raw = list_complete[0]    
    #print(date_begin_raw)
    year = int(str(date_begin_raw)[:4])
    month = int(str(date_begin_raw)[4:6])
    day = int(str(date_begin_raw)[6:8])
    #print(year, month, day)
    date_begin = date(year, month, day)
    #print(date_begin)
    numdays = len(prec_list)
    date_list = [date_begin + timedelta(days=x) for x in range(numdays)]
    #print(date_list)
    dict_ = {'Date': date_list,
             'Precipitation': prec_list}
    df_new = pd.DataFrame(dict_)
    df_new['Year'] = pd.DatetimeIndex(df_new['Date']).year
    df_new['Month'] = pd.DatetimeIndex(df_new['Date']).month
    df_new['Day'] = pd.DatetimeIndex(df_new['Date']).day
    
    if scenario == 'rcp 4.5':
        scenario_2 = 'rcp45'
    elif scenario == 'rcp 8.5':
        scenario_2 = 'rcp85'
    else:
        scenario_2 = scenario

    df_new.to_csv('GCM_data/bias_correction/{g}_{s}_{bc}_daily.csv'.format(g = name_gcm, s = scenario_2, bc = bias_correction_method), index = False)