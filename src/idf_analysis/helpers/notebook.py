import pandas as pd

def precip_summary(df, name):
    # Criar coluna de data
    if 'Hour' in df.columns:
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    else:
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    
    df = df.set_index('Date')
    
    non_zero = df[df['Precipitation'] > 0]
    
    stats = {
        "Total registros": len(df),
        "Registros > 0": len(non_zero),
        "Primeira data": df.index.min(),
        "Última data": df.index.max(),
        "Precipitação média (mm)": df['Precipitation'].mean(),
        "Precipitação máxima (mm)": df['Precipitation'].max(),
    }
    
    return pd.DataFrame(stats, index=[name])