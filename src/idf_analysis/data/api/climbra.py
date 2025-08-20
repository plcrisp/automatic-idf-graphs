from ...helpers.climbra_drive import extract_climbra_from_drive

from importlib import resources
from geopy.geocoders import Nominatim

import questionary
import pandas as pd
import numpy as np
import time



def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calcula a distância entre dois pontos usando a fórmula de Haversine.
    
    Args:
        lat1, lon1: Coordenadas do primeiro ponto
        lat2, lon2: Coordenadas do segundo ponto
    
    Returns:
        float: Distância em quilômetros
    """
    R = 6371  # Raio da Terra em km
    
    # Converte graus para radianos
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Diferenças
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Fórmula de Haversine
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c



def load_cabra_catchments() -> pd.DataFrame:
    """
    Loads the INMET station parameter data from the package resources.

    This function safely accesses the 'inmet_stations.csv' file that is
    bundled with the installed package.

    Returns:
        pd.DataFrame: A DataFrame containing the station parameters.
    """
    try:
        with resources.files('idf_analysis.resources').joinpath('cabra_catchments.csv').open('r', encoding='utf-8') as f:
            df = pd.read_csv(
                f,
                sep=',',
                on_bad_lines='warn' 
            )
        return df
    except FileNotFoundError:
        print("Error: Could not find the station parameters file within the package.")
        return pd.DataFrame()



def get_city_coordinates(city_input):
    # Cria o geolocalizador com um user_agent único
    geolocator = Nominatim(user_agent="idf_analysis_v1.0")
    
    try:
        # Adiciona "Brazil" para melhorar a busca
        search_query = f"{city_input}, Brazil"
        location = geolocator.geocode(search_query, timeout=10)
        
        if location:
            return location.latitude, location.longitude, location.address
        else:
            # Tenta sem "Brazil"
            location = geolocator.geocode(city_input, timeout=10)
            if location:
                return location.latitude, location.longitude, location.address
    
    except Exception as e:
        print(f"Erro na geocodificação: {e}")
    
    return None, None, None
    


def find_nearest_catchments(user_lat, user_lon, df_catchments, top_n=5):
    """
    Encontra os catchments mais próximos de uma localização.
    
    Args:
        user_lat (float): Latitude do usuário
        user_lon (float): Longitude do usuário  
        df_catchments (pd.DataFrame): DataFrame com os catchments
        top_n (int): Número de catchments mais próximos para retornar
        
    Returns:
        pd.DataFrame: DataFrame com os catchments mais próximos ordenados por distância
    """
    # Calcula distâncias
    distances = []
    for _, row in df_catchments.iterrows():
        dist = calculate_distance(user_lat, user_lon, row['latitude'], row['longitude'])
        distances.append(dist)
    
    df_with_distances = df_catchments.copy()
    df_with_distances['distance_km'] = distances
    
    # Ordena por distância e retorna os top_n
    return df_with_distances.sort_values('distance_km').head(top_n)

def get_climbra_data():
    """
    Função principal para obter dados de precipitação do dataset climbra.
    
    Processo:
        1. Carrega dados dos catchments CABra
        2. Interface interativa para escolha de cidade
        3. Geocoding da cidade escolhida
        4. Seleção do catchment mais próximo
        5. Escolha do tipo de dado (histórico, SSP245, SSP585)
        6. Seleção do período de dados
        7. Extração dos dados de precipitação
        8. Dados de precipitação no formato necessário
    
    Returns:
        dict: Dicionário com informações da seleção feita
    """
    
    print("🌧️  === SISTEMA DE OBTENÇÃO DE DADOS CABra ===")
    print("📊 Carregando dados dos catchments...")
    
    # 1. Carregar dados dos catchments
    df_catchments = load_cabra_catchments()
    
    if df_catchments.empty:
        print("❌ Não foi possível carregar os dados dos catchments.")
        return None
    
    print(f"✅ {len(df_catchments)} catchments carregados!")
    print(f"📍 Cobertura: {df_catchments['gauge_state'].nunique()} estados brasileiros")
    
    # 2. Seleção da cidade
    print("\n🏙️  === SELEÇÃO DE LOCALIDADE ===")
    
    # Opções de entrada de localidade
    method = questionary.select(
        "Como você quer informar sua localidade?",
        choices=[
            "🔍 Buscar por nome da cidade",
            "📍 Informar coordenadas manualmente"
        ]
    ).ask()
    
    user_lat, user_lon, city_address = None, None, None
    
    if "Buscar por nome" in method:
        while True:
            city_name = questionary.text(
                "Digite o nome da cidade:"
            ).ask()
            
            if not city_name:
                continue
                
            print(f"🔍 Buscando coordenadas para '{city_name}'...")
            user_lat, user_lon, city_address = get_city_coordinates(city_name)
            
            if user_lat and user_lon:
                print(f"✅ Cidade encontrada!")
                print(f"📍 Coordenadas: {user_lat:.4f}, {user_lon:.4f}")
                print(f"🏠 Endereço: {city_address}")
                
                confirm = questionary.confirm("Confirma esta localização?").ask()
                if confirm:
                    break
            else:
                print("❌ Cidade não encontrada. Tente outra grafia ou seja mais específico.")
                retry = questionary.confirm("Tentar novamente?").ask()
                if not retry:
                    return None
            
            time.sleep(1)  # Respeita limite do Nominatim
    
    else:  # Coordenadas manuais
        try:
            user_lat = float(questionary.text("Latitude:").ask())
            user_lon = float(questionary.text("Longitude:").ask())
            city_address = f"Coordenadas: {user_lat:.4f}, {user_lon:.4f}"
        except ValueError:
            print("❌ Coordenadas inválidas.")
            return None
    
    # 3. Encontrar catchments mais próximos
    print("\n📡 === BUSCA DE CATCHMENTS PRÓXIMOS ===")
    print("🔍 Calculando distâncias...")
    
    nearest_catchments = find_nearest_catchments(user_lat, user_lon, df_catchments, top_n=10)
    
    print(f"✅ Encontrados {len(nearest_catchments)} catchments próximos:")
    print("\n🏆 TOP 5 MAIS PRÓXIMOS:")
    for i, (_, row) in enumerate(nearest_catchments.head().iterrows(), 1):
        print(f"{i}. CABra_{row['CABra_ID']} - {row['distance_km']:.1f}km")
        print(f"   📍 {row['gauge_state']} | {row['gauge_biome']} | Qualidade: {row['quality_index']:.1f}%")
    
    # 4. Seleção do catchment
    catchment_choices = []
    for _, row in nearest_catchments.head().iterrows():
        choice_text = (f"CABra_{row['CABra_ID']} - {row['distance_km']:.1f}km "
                      f"({row['gauge_state']}, {row['gauge_biome']}, "
                      f"Qual: {row['quality_index']:.1f}%)")
        catchment_choices.append(choice_text)
    
    selected_catchment_str = questionary.select(
        "Escolha o catchment para análise:",
        choices=catchment_choices
    ).ask()
    
    # Extrair ID do catchment selecionado
    selected_id = int(selected_catchment_str.split("CABra_")[1].split(" ")[0])
    selected_catchment = nearest_catchments[nearest_catchments['CABra_ID'] == selected_id].iloc[0]
    
    print(f"\n✅ Catchment selecionado: CABra_{selected_id}")
    print(f"📊 Distância: {selected_catchment['distance_km']:.1f}km")
    print(f"🗺️  Estado: {selected_catchment['gauge_state']}")
    print(f"🌿 Bioma: {selected_catchment['gauge_biome']}")
    print(f"⭐ Qualidade: {selected_catchment['quality_index']:.1f}%")
    
    # 5. Seleção do tipo de dados
    print("\n📈 === TIPO DE DADOS ===")
    
    data_type = questionary.select(
        "Escolha o tipo de dados:",
        choices=[
            "📚 Dados Históricos (1980-2013)",
            "🌡️  Projeção SSP245 (2015-2100, cenário otimista)",
            "🔥 Projeção SSP585 (2015-2100, cenário pessimista)"
        ]
    ).ask()
    
    # Mapear escolha para código
    if "Históricos" in data_type:
        scenario = "historical"
        period_start = 1980
        period_end = 2013
    elif "SSP245" in data_type:
        scenario = "ssp245"
        period_start = 2015
        period_end = 2100
    else:  # SSP585
        scenario = "ssp585"
        period_start = 2015
        period_end = 2100
    
    print(f"✅ Tipo selecionado: {scenario.upper()}")
    print(f"📅 Período disponível: {period_start}-{period_end}")
    
    # 6. Seleção do período específico
    print(f"\n📅 === PERÍODO DE ANÁLISE ===")
    
    period_choice = questionary.select(
        "Escolha o período:",
        choices=[
            f"📊 Período completo ({period_start}-{period_end})",
            "🎯 Período personalizado"
        ]
    ).ask()
    
    if "personalizado" in period_choice:
        while True:
            try:
                start_year = int(questionary.text(
                    f"Ano inicial (>= {period_start}):"
                ).ask())
                end_year = int(questionary.text(
                    f"Ano final (<= {period_end}):"
                ).ask())
                
                if start_year < period_start or end_year > period_end or start_year > end_year:
                    print(f"❌ Anos devem estar entre {period_start}-{period_end} e ano inicial ≤ final")
                    continue
                    
                break
            except ValueError:
                print("❌ Digite anos válidos (números inteiros)")
    else:
        start_year = period_start
        end_year = period_end
    
    # 7. Resumo da seleção
    print("\n" + "="*60)
    print("📋 === RESUMO DA SELEÇÃO ===")
    print("="*60)
    print(f"🏙️  Localidade: {city_address}")
    print(f"📍 Coordenadas: {user_lat:.4f}, {user_lon:.4f}")
    print(f"🎯 Catchment: CABra_{selected_id}")
    print(f"📏 Distância: {selected_catchment['distance_km']:.1f}km")
    print(f"🗺️  Estado/Bioma: {selected_catchment['gauge_state']}/{selected_catchment['gauge_biome']}")
    print(f"📈 Cenário: {scenario.upper()}")
    print(f"📅 Período: {start_year}-{end_year}")
    print("="*60)
    
    confirm_selection = questionary.confirm(
        "Confirma a seleção acima para obter os dados?"
    ).ask()
    
    if not confirm_selection:
        print("❌ Operação cancelada pelo usuário.")
        return None
    
    # 8. Preparar informações para extração de dados
    selection_info = {
        'user_location': {
            'latitude': user_lat,
            'longitude': user_lon,
            'address': city_address
        },
        'selected_catchment': {
            'id': selected_id,
            'latitude': selected_catchment['latitude'],
            'longitude': selected_catchment['longitude'],
            'state': selected_catchment['gauge_state'],
            'biome': selected_catchment['gauge_biome'],
            'quality_index': selected_catchment['quality_index'],
            'distance_km': selected_catchment['distance_km']
        },
        'data_request': {
            'scenario': scenario,
            'start_year': start_year,
            'end_year': end_year,
            'column_name': f'CABra_{selected_id}'
        }
    }
    
    try:
        drive_folder = "1L5rNsn9uMVVKrteuuDGrci92mThY3IFq"  

        df_precip, saved_path = extract_climbra_from_drive(
            selection_info=selection_info,
            drive_folder_url_or_id=drive_folder,
            credentials_json_path=None,        
            cache_dir="./.cache/climbra"       
        )

        selection_info["result"] = {
            "rows": len(df_precip),
            "saved_csv": saved_path,
            "columns": list(df_precip.columns),
        }

        print(f"✅ Dados prontos! {len(df_precip)} linhas → {saved_path}")

    except Exception as e:
        print(f"❌ Falha ao extrair dados do Drive: {e}")
        return None

    return selection_info