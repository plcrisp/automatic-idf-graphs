#from ...helpers.climbra_drive import extract_climbra_from_drive

from importlib import resources
from geopy.geocoders import Nominatim

import questionary
import pandas as pd
import numpy as np
import time
import unicodedata
from urllib.parse import urlparse, parse_qs, unquote
import requests
import math
from pathlib import Path
import xarray as xr

CLIMBRA_DATASETS_INDEX_RESOURCE = 'climbra_datasets.txt'


class AllowedRootFolders:
    """
    Constantes para pastas raiz permitidas no navegador CLIMBra.
    
    Usage:
        >>> choose_and_download_climbra_dataset(
        ...     allowed_roots=[AllowedRootFolders.GriddedData, AllowedRootFolders.ETo]
        ... )
    """
    CatchmentsDataV3 = "Catchments-Data-v3"
    GriddedData = "Gridded data"
    EnsembleData = "Ensemble data"
    ETo = "ETo"
    
    @classmethod
    def all(cls) -> list[str]:
        """Retorna todas as pastas raiz disponíveis."""
        return [cls.CatchmentsDataV3, cls.GriddedData, cls.EnsembleData, cls.ETo]
    
    @classmethod
    def default(cls) -> list[str]:
        """Retorna a lista padrão de pastas raiz."""
        return [cls.CatchmentsDataV3, cls.GriddedData]


class AllowedExtraFiles:
    """
    Constantes para arquivos extras sempre incluídos no navegador CLIMBra.
    
    Usage:
        >>> choose_and_download_climbra_dataset(
        ...     allowed_extra_files=[AllowedExtraFiles.ReadMe]
        ... )
    """
    ReadMe = "READ_ME_paper2.docx"
    
    @classmethod
    def all(cls) -> list[str]:
        """Retorna todos os arquivos extras disponíveis."""
        return [cls.ReadMe]
    
    @classmethod
    def default(cls) -> list[str]:
        """Retorna a lista padrão de arquivos extras."""
        return [] # Nenhum por padrão


def _sanitize_city_name(city_name: str) -> str:
    """
    Remove acentos e caracteres especiais de nomes de cidades para uso em nomes de arquivos.
    
    Args:
        city_name: Nome da cidade com possíveis acentos e caracteres especiais
        
    Returns:
        str: Nome sanitizado (sem acentos, minúsculas, espaços substituídos por underscore)
    """
    # Remove acentos usando normalização NFD (decomposição)
    nfd = unicodedata.normalize('NFD', city_name)
    # Filtra apenas caracteres ASCII (remove diacríticos)
    without_accents = ''.join(char for char in nfd if unicodedata.category(char) != 'Mn')
    # Remove vírgulas, pontos e outros caracteres especiais, substitui por espaços
    cleaned = without_accents.replace(',', '').replace('.', '').replace('-', ' ')
    # Converte para minúsculas e substitui espaços por underscore
    sanitized = cleaned.lower().strip().replace(' ', '_')
    # Remove múltiplos underscores consecutivos
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    return sanitized


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
    [DEPRECATED]
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

'''

def get_climbra_data():
    """
    [DEPRECATED]
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
'''

# ===============================
# Navegação de datasets
# ===============================

def _read_dataset_index_from_resource() -> list[str]:
    """
    Lê o arquivo de índice (TXT) com URLs de datasets do CLIMBra a partir dos recursos do pacote.

    Returns:
        list[str]: Lista de URLs (uma por linha) sem linhas vazias/comentários.
    """
    resource_name = CLIMBRA_DATASETS_INDEX_RESOURCE
    try:
        with resources.files('idf_analysis.resources').joinpath(resource_name).open('r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines()]
        # Filtra vazias e comentários
        lines = [ln for ln in lines if ln and not ln.lstrip().startswith('#')]
        return lines
    except FileNotFoundError:
        print(f"❌ Arquivo de índice '{resource_name}' não encontrado em idf_analysis.resources")
        return []


def _parse_dataset_entry(url: str) -> dict:
    """
    Faz o parse de uma URL do repositório e extrai caminho, nome do arquivo e outros metadados.

    A URL esperada possui query params `path` (diretório no servidor) e `fileName` (nome do arquivo).

    Returns:
        dict: { 'url', 'file_name', 'path', 'segments', 'full_path' }
    """
    try:
        pr = urlparse(url)
        qs = parse_qs(pr.query)
        p = unquote(qs.get('path', [''])[0])  # ex: /V5/Gridded data/pr/ssp585/...
        file_name = unquote(qs.get('fileName', [''])[0])
        # Normaliza e quebra em segmentos (ignora vazio inicial e opcional 'V5')
        parts = [seg for seg in p.split('/') if seg]
        if parts and parts[0] == 'V5':
            parts = parts[1:]
        # Remoção de nível redundante: alguns índices possuem o último segmento
        # igual ao nome do arquivo (ex: .../MIROC6-pr-hist.nc + fileName=MIROC6-pr-hist.nc),
        # gerando uma pasta desnecessária com o mesmo nome do arquivo
        if file_name and parts and parts[-1] == file_name:
            parts = parts[:-1]
        full_path = '/'.join(parts + ([file_name] if file_name else []))
        return {
            'url': url,
            'file_name': file_name or url.split('/')[-1],
            'path': p,
            'segments': parts,
            'full_path': full_path,
        }
    except Exception:
        # Em caso de formato inesperado, retorna mínimos
        return {
            'url': url,
            'file_name': url.split('/')[-1],
            'path': '',
            'segments': [],
            'full_path': url,
        }


def _build_tree(entries: list[dict]) -> dict:
    """
    Constrói uma árvore de navegação (pastas/arquivos) a partir das entradas parseadas.

    Estrutura do nó:
      {
        '_files': [ {'name': str, 'url': str, 'full_path': str} ],
        '<folder>': { ...subtree... }
      }
    """
    root: dict = {'_files': []}
    for e in entries:
        node = root
        for seg in e['segments']:
            node = node.setdefault(seg, {'_files': []})
        node['_files'].append({'name': e['file_name'], 'url': e['url'], 'full_path': e['full_path']})
    return root


def _list_node(node: dict) -> tuple[list[str], list[dict]]:
    """Retorna (subpastas, arquivos) do nó atual, ordenados."""
    folders = sorted([k for k in node.keys() if k != '_files'])
    files = sorted(node.get('_files', []), key=lambda x: x['name'])
    return folders, files


def _search_files(root: dict, query: str) -> list[dict]:
    """Busca arquivos por substring em nome ou caminho completo."""
    results = []

    def dfs(node: dict, path: list[str]):
        for f in node.get('_files', []):
            full = '/'.join(path + [f['name']])
            if query.lower() in f['name'].lower() or query.lower() in full.lower():
                results.append({'display': full, **f})
        for k, v in node.items():
            if k == '_files':
                continue
            dfs(v, path + [k])

    dfs(root, [])
    # Ordena por caminho
    results.sort(key=lambda x: x['display'])
    return results


def choose_climbra_dataset_url(
    allowed_roots: list[str] | None = None,
    allowed_extra_files: list[str] | None = None,
) -> str | None:
    """
    Abre um navegador interativo no terminal para selecionar um dataset do CLIMBra
    a partir do índice TXT empacotado no pacote. Ao final, retorna a URL selecionada.

    Fluxo:
      1) Lê o arquivo TXT em idf_analysis.resources (uma URL por linha)
      2) Constrói uma árvore de pastas baseada no parâmetro `path` de cada URL
      3) Permite navegar por pastas, buscar por nome e selecionar um arquivo

    Args:
        allowed_roots: Lista de pastas raiz que serão exibidas (default: AllowedRootFolders.default())
        allowed_extra_files: Lista de nomes de arquivos a sempre incluir (default: AllowedExtraFiles.default())

    Returns:
        str | None: URL selecionada ou None se o usuário cancelar.
    """
    urls = _read_dataset_index_from_resource()
    if not urls:
        return None

    allowed_roots = allowed_roots if allowed_roots is not None else AllowedRootFolders.default()
    allowed_extra_files = allowed_extra_files if allowed_extra_files is not None else AllowedExtraFiles.default()

    entries_all = [_parse_dataset_entry(u) for u in urls]

    # Filtra por pastas raiz permitidas OU arquivos extras explicitamente permitidos
    entries = []
    for e in entries_all:
        root = e['segments'][0] if e['segments'] else ''
        if root in allowed_roots or e['file_name'] in allowed_extra_files:
            entries.append(e)

    tree = _build_tree(entries)

    # Navegação
    path_stack: list[tuple[str, dict]] = [('root', tree)]

    while True:
        current_name, current_node = path_stack[-1]
        folders, files = _list_node(current_node)

        breadcrumb = ' / '.join([p[0] for p in path_stack])
        choices = []
        # Pastas
        for d in folders:
            choices.append(questionary.Choice(title=f"📁 {d}", value=("dir", d)))
        # Arquivos
        for f in files:
            choices.append(questionary.Choice(title=f"📄 {f['name']}", value=("file", f)))

        # Ações especiais
        actions = [
            questionary.Choice(title="🔎 Buscar por nome...", value=("search", None)),
        ]
        if len(path_stack) > 1:
            actions.insert(0, questionary.Choice(title="⬆️  Voltar", value=("up", None)))
        actions.append(questionary.Choice(title="❌ Cancelar", value=("cancel", None)))

        answer = questionary.select(
            message=f"Selecione ( {breadcrumb} )",
            choices=choices + [questionary.Separator("-"*24)] + actions,
            qmark="🌐"
        ).ask()

        if answer is None:
            return None

        kind, payload = answer
        if kind == 'dir':
            # Entra na subpasta
            path_stack.append((payload, current_node[payload]))
            continue
        elif kind == 'up':
            if len(path_stack) > 1:
                path_stack.pop()
            continue
        elif kind == 'file':
            # Retorna URL diretamente
            return payload['url']
        elif kind == 'search':
            query = questionary.text("Digite parte do nome do arquivo (ex: pr-ssp585, .nc, .csv):").ask()
            if not query:
                continue
            results = _search_files(tree, query)
            if not results:
                print("🔍 Nenhum resultado encontrado.")
                continue
            sel = questionary.select(
                message=f"Resultados para '{query}':",
                choices=[questionary.Choice(title=f"📄 {r['display']}", value=r) for r in results] + [questionary.Choice(title="⬅️  Voltar", value=None)],
                qmark="🔎",
            ).ask()
            if sel is None:
                continue
            # Retorna URL diretamente
            return sel['url']
        elif kind == 'cancel':
            return None

    # Fallback
    return None


def _derive_filename_from_url(url: str) -> str:
    """Tenta derivar o nome do arquivo a partir dos parâmetros da URL ou últimos segmentos."""
    pr = urlparse(url)
    qs = parse_qs(pr.query)
    file_name = qs.get('fileName', [None])[0]
    if file_name:
        return unquote(file_name)
    # fallback usando path
    tail = pr.path.rstrip('/').split('/')[-1]
    return tail or 'download.bin'


def download_climbra_dataset(
    url: str,
    output_dir: str = './downloads/climbra',
    chunk_size: int = 1 << 14,
    max_retries: int = 3,
    timeout: int = 30,
    show_progress: bool = True,
    destination_path: Path | None = None,
) -> Path | None:
    """
    Faz download streaming de um dataset CLIMBra com barras de progresso e estimativa de tempo.

    Args:
        url: URL completa do arquivo.
        output_dir: Diretório onde salvar.
        chunk_size: Tamanho do bloco (bytes) para leitura streaming.
        max_retries: Número máximo de tentativas em falha transitória.
        timeout: Timeout da requisição (segundos).
        show_progress: Se True, imprime progresso percentual, velocidade e tempo estimado.

    Returns:
        Path | None: Caminho do arquivo salvo ou None em caso de falha.
    """
    if not url:
        print('❌ URL vazia fornecida.')
        return None

    if destination_path is not None:
        out_path = Path(destination_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        filename = _derive_filename_from_url(url)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            print(f"\n🌐 Download (tentativa {attempt}/{max_retries}): {url}")
            with requests.get(url, stream=True, timeout=timeout) as resp:
                status = resp.status_code
                if status != 200:
                    print(f"❌ HTTP {status}: Falha ao iniciar download")
                    continue
                total = resp.headers.get('Content-Length')
                total_bytes = int(total) if total and total.isdigit() else None
                received = 0
                start_time = time.time()
                speed_samples = []

                with open(out_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        f.write(chunk)
                        received += len(chunk)
                        if show_progress:
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed = received / elapsed  # bytes/s
                                speed_samples.append(speed)
                            if total_bytes:
                                pct = received / total_bytes * 100
                                bar_len = 40
                                filled = int(bar_len * pct / 100)
                                bar = '█' * filled + '░' * (bar_len - filled)
                                human_total = _human_readable_size(total_bytes)
                                human_recv = _human_readable_size(received)
                                
                                # Estimativa de tempo restante
                                eta_str = ''
                                if speed > 0 and pct > 0:
                                    remaining_bytes = total_bytes - received
                                    eta_seconds = remaining_bytes / speed
                                    if eta_seconds < 60:
                                        eta_str = f" | ETA: {int(eta_seconds)}s"
                                    elif eta_seconds < 3600:
                                        eta_str = f" | ETA: {int(eta_seconds/60)}m {int(eta_seconds%60)}s"
                                    else:
                                        eta_str = f" | ETA: {int(eta_seconds/3600)}h {int((eta_seconds%3600)/60)}m"
                                
                                status = (
                                    f"⬇️  [{bar}] {pct:6.2f}% {human_recv}/{human_total}  "
                                    f"{_human_readable_size(int(speed))}/s{eta_str}"
                                )
                                print(f"\r{status}\x1b[K", end='', flush=True)
                            else:
                                human_recv = _human_readable_size(received)
                                print(f"\r⬇️  {human_recv} recebidos...\x1b[K", end='', flush=True)
                if show_progress:
                    print()  # nova linha
                print(f"✅ Download concluído: {out_path} ({_human_readable_size(received)})")
                return out_path
        except requests.RequestException as e:
            print(f"⚠️  Erro de rede: {e}")
        except Exception as e:
            print(f"⚠️  Erro inesperado: {e}")
        print("🔁 Re-tentando...")

    print("❌ Todas as tentativas de download falharam.")
    return None


def _human_readable_size(n_bytes: int) -> str:
    """Converte bytes em formato legível (KB/MB/GB)."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    exp = int(math.log(n_bytes, 1024))
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    exp = min(exp, len(units) - 1)
    value = n_bytes / (1024 ** exp)
    return f"{value:.2f} {units[exp]}"


def _ensure_xarray_available():
    if xr is None:
        raise RuntimeError(
            "xarray não está disponível. Instale as dependências: pip install xarray netcdf4"
        )


def netcdf_precipitation_to_csv(
    nc_path: str | Path,
    csv_path: str | Path | None = None,
    subset: str = "area",
    lat_init: float | None = None,
    lat_fin: float | None = None,
    lon_init: float | None = None,
    lon_fin: float | None = None,
    lat: float | None = None,
    lon: float | None = None,
    variable: str = "pr",
    start_year: int | None = None,
    end_year: int | None = None,
) -> Path:
    """
    Converte um arquivo NetCDF (variável 'pr') em uma série diária e exporta CSV
    com colunas Year, Month, Day, Precipitation (mm/day).

    Args:
        nc_path: Caminho do arquivo .nc local.
        csv_path: Caminho de saída do CSV. Se None, gera ao lado do .nc.
        subset: 'area' (média espacial) ou 'point' (ponto mais próximo).
        lat_init, lat_fin, lon_init, lon_fin: limites da área (quando subset='area').
        lat, lon: coordenadas do ponto (quando subset='point').
        variable: nome da variável de precipitação (default: 'pr').
        start_year: ano inicial para filtrar os dados (opcional).
        end_year: ano final para filtrar os dados (opcional).

    Returns:
        Path: caminho do CSV gerado.
    """
    _ensure_xarray_available()
    nc_path = Path(nc_path)
    ds = xr.open_dataset(nc_path)

    if variable not in ds.variables:
        # tenta detectar automaticamente
        if "pr" in ds.variables:
            variable = "pr"
        else:
            raise ValueError(f"Variável '{variable}' não encontrada no dataset. Disponíveis: {list(ds.variables)}")

    da = ds[variable]

    # Subset
    if subset.lower() == "area":
        if None in (lat_init, lat_fin, lon_init, lon_fin):
            raise ValueError("Para subset='area', informe lat_init, lat_fin, lon_init, lon_fin.")
        # Garante ordem adequada nos slices
        lat0, lat1 = (lat_init, lat_fin) if (lat_init <= lat_fin) else (lat_fin, lat_init)
        lon0, lon1 = (lon_init, lon_fin) if (lon_init <= lon_fin) else (lon_fin, lon_init)
        sub = da.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
        ts = sub.mean(dim=[d for d in sub.dims if d != 'time'])
    elif subset.lower() == "point":
        if lat is None or lon is None:
            raise ValueError("Para subset='point', informe lat e lon.")
        ts = da.sel(lat=lat, lon=lon, method="nearest")
    else:
        raise ValueError("subset deve ser 'area' ou 'point'.")

    # Converte para DataFrame
    df = ts.to_series().reset_index()
    # Renomeia a coluna de tempo se necessário
    if 'time' not in df.columns:
        # tenta deduzir a primeira coluna temporal
        if np.issubdtype(df.columns[0].dtype, np.datetime64):
            df = df.rename(columns={df.columns[0]: 'time'})
        else:
            df = df.rename(columns={df.columns[0]: 'time'})

    # Conversão de unidades para mm/day quando necessário
    units = str(da.attrs.get('units', '')).lower()
    precip = df[variable].astype(float)
    if 'kg' in units and 'm-2' in units and ('s-1' in units or 's^-1' in units):
        # 1 kg m-2 s-1 = 1 mm/s; diário → * 86400
        precip = precip * 86400.0
    # Se já estiver em mm/day, mantém

    dt = pd.to_datetime(df['time']).dt
    out = pd.DataFrame({
        'Year': dt.year,
        'Month': dt.month,
        'Day': dt.day,
        'Precipitation': precip.values,
    })

    # Filtrar por período se especificado
    if start_year is not None:
        out = out[out['Year'] >= start_year]
    if end_year is not None:
        out = out[out['Year'] <= end_year]

    # Define caminho de saída
    if csv_path is None:
        stem = nc_path.stem
        suffix = '_area' if subset.lower() == 'area' else '_point'
        csv_path = nc_path.with_name(f"{stem}{suffix}.csv")

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    return csv_path


def _get_netcdf_grid_points(nc_path: str | Path, variable: str = "pr") -> pd.DataFrame:
    """
    Extrai todos os pontos de grid (lat, lon) de um arquivo NetCDF.
    
    Args:
        nc_path: Caminho do arquivo .nc
        variable: Nome da variável para detectar dimensões
        
    Returns:
        pd.DataFrame com colunas 'latitude', 'longitude', 'grid_index'
    """
    _ensure_xarray_available()
    ds = xr.open_dataset(nc_path)
    
    if variable not in ds.variables:
        if "pr" in ds.variables:
            variable = "pr"
        else:
            raise ValueError(f"Variável '{variable}' não encontrada")
    
    da = ds[variable]
    
    # Detectar dimensões de lat/lon
    lat_dim = [d for d in da.dims if d in ['lat', 'latitude', 'y']]
    lon_dim = [d for d in da.dims if d in ['lon', 'longitude', 'x']]
    
    if not lat_dim or not lon_dim:
        raise ValueError("Não foi possível detectar dimensões de latitude/longitude no NetCDF")
    
    lats = ds[lat_dim[0]].values
    lons = ds[lon_dim[0]].values
    
    # Criar grid de pontos
    grid_points = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            grid_points.append({
                'latitude': float(lat),
                'longitude': float(lon),
                'grid_index': f"lat{i}_lon{j}",
                'lat_idx': i,
                'lon_idx': j,
            })
    
    ds.close()
    return pd.DataFrame(grid_points)


def process_local_netcdf_with_city(
    nc_path: str | Path,
    csv_path: str | Path | None = None,
    city_name: str | None = None,
    variable: str = "pr",
    top_n: int = 10,
    output_dir: str = './results/climbra',
    results_subdir_segments: list[str] | None = None,
    allow_filename_edit: bool = True,
) -> Path | None:
    """
    Processa um arquivo NetCDF já baixado e gera CSV com busca de coordenadas por nome da cidade.
    
    Fluxo:
        1. Se city_name fornecido, busca coordenadas via geocoding (Nominatim)
        2. Carrega pontos de grid do NetCDF
        3. Calcula distâncias e mostra pontos mais próximos
        4. Usuário seleciona um ou mais pontos:
           - 1 ponto: extração pontual (subset='point')
           - 2+ pontos: média espacial (subset='area')
        5. Seleciona período de análise
        6. Gera CSV (Year, Month, Day, Precipitation)
    
    Args:
        nc_path: Caminho do arquivo .nc local.
        csv_path: Caminho de saída do CSV. Se None, gera automaticamente em output_dir.
        city_name: Nome da cidade para buscar coordenadas (ex: "São Paulo, SP").
        variable: Nome da variável de precipitação (default: 'pr').
        top_n: Número de pontos mais próximos a mostrar (default: 10).
        output_dir: Diretório onde salvar o CSV (default: './results/climbra').
    
    Returns:
        Path | None: Caminho do CSV gerado ou None em caso de falha.
    """
    _ensure_xarray_available()
    
    print("🌐 === PROCESSAMENTO DE DATASET CLIMBra LOCAL ===")
    print(f"📂 Arquivo: {nc_path}")
    
    # 1. Buscar coordenadas da cidade
    print("\n🏙️  === SELEÇÃO DE LOCALIDADE ===")
    
    if city_name:
        print(f"🔍 Buscando coordenadas para '{city_name}'...")
        user_lat, user_lon, city_address = get_city_coordinates(city_name)
        
        if user_lat and user_lon:
            print(f"✅ Cidade encontrada!")
            print(f"📍 Coordenadas: {user_lat:.4f}, {user_lon:.4f}")
            print(f"🏠 Endereço: {city_address}")
        else:
            print("⚠️  Cidade não encontrada.")
            city_address = None
            # Fallback para entrada manual
            try:
                user_lat = float(questionary.text("Latitude:").ask())
                user_lon = float(questionary.text("Longitude:").ask())
                city_address = f"Coordenadas: {user_lat:.4f}, {user_lon:.4f}"
            except (TypeError, ValueError):
                print("❌ Coordenadas inválidas.")
                return None
    else:
        # Entrada manual
        try:
            user_lat = float(questionary.text("Latitude:").ask())
            user_lon = float(questionary.text("Longitude:").ask())
            city_address = f"Coordenadas: {user_lat:.4f}, {user_lon:.4f}"
        except (TypeError, ValueError):
            print("❌ Coordenadas inválidas.")
            return None
    
    # 2. Carregar pontos de grid do NetCDF
    print("\n📡 === CARREGANDO GRID DO NETCDF ===")
    print("🔍 Lendo pontos de grid...")
    
    try:
        df_grid = _get_netcdf_grid_points(nc_path, variable=variable)
        print(f"✅ {len(df_grid)} pontos de grid carregados")
    except Exception as e:
        print(f"❌ Erro ao carregar grid do NetCDF: {e}")
        return None
    
    # 3. Calcular distâncias e encontrar pontos mais próximos
    print("\n📊 === BUSCA DE PONTOS MAIS PRÓXIMOS ===")
    print("🔍 Calculando distâncias...")
    
    nearest_points = find_nearest_catchments(user_lat, user_lon, df_grid, top_n=top_n)
    
    # 4. Seleção de pontos
    print("\n🎯 === SELEÇÃO DE PONTOS ===")
    
    point_choices = []
    points_dict_list = []
    for _, row in nearest_points.iterrows():
        # Converter Series para dict para evitar problemas com questionary
        point_dict = {
            'grid_index': row['grid_index'],
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'distance_km': row['distance_km'],
            'lat_idx': row['lat_idx'],
            'lon_idx': row['lon_idx'],
        }
        points_dict_list.append(point_dict)
        
        choice_text = (f"📍 Grid {point_dict['grid_index']} - {point_dict['distance_km']:.2f}km "
                      f"(Lat: {point_dict['latitude']:.4f}, Lon: {point_dict['longitude']:.4f})")
        point_choices.append(questionary.Choice(title=choice_text, value=len(points_dict_list) - 1))
    
    selected_indices = questionary.checkbox(
        "Selecione um ou mais pontos (use ESPAÇO para marcar, ENTER para confirmar):",
        choices=point_choices
    ).ask()
    
    if not selected_indices:
        print("❌ Nenhum ponto selecionado.")
        return None
    
    # Recuperar os pontos selecionados usando os índices
    selected_points = [points_dict_list[i] for i in selected_indices]
    
    # 5. Determinar subset automaticamente
    if len(selected_points) == 1:
        subset = 'point'
        point = selected_points[0]
        params = {
            'lat': point['latitude'],
            'lon': point['longitude'],
        }
        print(f"\n📍 1 ponto selecionado → Extração PONTUAL")
        print(f"   Grid: {point['grid_index']} ({point['distance_km']:.2f}km)")
        print(f"   Coordenadas: {point['latitude']:.4f}, {point['longitude']:.4f}")
    else:
        subset = 'area'
        lats = [p['latitude'] for p in selected_points]
        lons = [p['longitude'] for p in selected_points]
        params = {
            'lat_init': min(lats),
            'lat_fin': max(lats),
            'lon_init': min(lons),
            'lon_fin': max(lons),
        }
        print(f"\n📦 {len(selected_points)} pontos selecionados → MÉDIA ESPACIAL")
        print(f"   Área de cobertura:")
        print(f"   • Latitude: [{params['lat_init']:.4f}, {params['lat_fin']:.4f}]")
        print(f"   • Longitude: [{params['lon_init']:.4f}, {params['lon_fin']:.4f}]")
        print(f"   Pontos incluídos:")
        for i, p in enumerate(selected_points, 1):
            print(f"   {i}. Grid {p['grid_index']} - {p['distance_km']:.2f}km")
    
    # 6. Selecionar período de análise
    print("\n📅 === PERÍODO DE ANÁLISE ===")
    
    # Detectar período disponível no NetCDF
    try:
        ds_temp = xr.open_dataset(nc_path)
        time_var = ds_temp['time']
        dates = pd.to_datetime(time_var.values)
        period_start = dates.min().year
        period_end = dates.max().year
        ds_temp.close()
        
        print(f"📊 Período disponível no dataset: {period_start}-{period_end}")
        
        period_choice = questionary.select(
            "Escolha o período:",
            choices=[
                f"📊 Período completo ({period_start}-{period_end})",
                "🎯 Período personalizado"
            ]
        ).ask()
        
        if period_choice is None:
            print("❌ Operação cancelada.")
            return None
        
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
                    
                    print(f"✅ Período selecionado: {start_year}-{end_year}")
                    break
                except (ValueError, TypeError):
                    print("❌ Digite anos válidos (números inteiros)")
        else:
            start_year = period_start
            end_year = period_end
            print(f"✅ Usando período completo: {start_year}-{end_year}")
            
    except Exception as e:
        print(f"⚠️  Não foi possível detectar o período: {e}")
        print("Continuando sem filtro de período...")
        start_year = None
        end_year = None
    
    # 7. Gerar nome e pasta do CSV se não fornecido (espelhando árvore e permitindo edição)
    if csv_path is None:
        base_dir = Path(output_dir)
        if results_subdir_segments:
            base_dir = base_dir.joinpath(*results_subdir_segments)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Nome base (cidade sanitizada quando disponível)
        if city_name:
            name_base = _sanitize_city_name(city_name)
        else:
            if subset == 'point' and 'lat' in locals() and 'lon' in locals():
                try:
                    name_base = f"lat{params['lat']:.2f}_lon{params['lon']:.2f}"
                except Exception:
                    name_base = 'serie'
            else:
                name_base = 'area'

        # Sufixo com período e subset
        if start_year is not None and end_year is not None:
            default_name = f"{name_base}_{start_year}-{end_year}_{subset}.csv"
        else:
            default_name = f"{name_base}_{subset}.csv"

        filename = default_name
        if allow_filename_edit:
            edit = questionary.text(
                "Nome do arquivo CSV:", default=default_name
            ).ask()
            if edit:
                # Remove .csv temporariamente se já estiver presente
                edit_clean = edit[:-4] if edit.endswith('.csv') else edit
                # Sanitiza novamente para evitar caracteres inválidos
                filename = _sanitize_city_name(edit_clean).replace('__', '_') + '.csv'
        csv_path = base_dir / filename
    
    # 8. Gerar CSV
    print(f"\n⚙️  Gerando CSV...")
    try:
        csv_result = netcdf_precipitation_to_csv(
            nc_path=nc_path,
            csv_path=csv_path,
            subset=subset,
            variable=variable,
            start_year=start_year,
            end_year=end_year,
            **params,
        )
        print(f"✅ CSV gerado com sucesso: {csv_result}")
        
        # 9. Resumo final
        print("\n" + "="*60)
        print("📋 === RESUMO ===")
        print("="*60)
        if city_address:
            print(f"🏙️  Localidade: {city_address}")
        print(f"📂 Arquivo NetCDF: {nc_path}")
        print(f"📊 Método: {subset.upper()}")
        print(f"🎯 Pontos selecionados: {len(selected_points)}")
        if subset == 'point':
            print(f"📍 Coordenadas: {params['lat']:.4f}, {params['lon']:.4f}")
            print(f"📏 Distância da localidade: {selected_points[0]['distance_km']:.2f}km")
        else:
            print(f"📦 Área: lat [{params['lat_init']:.4f}, {params['lat_fin']:.4f}], lon [{params['lon_init']:.4f}, {params['lon_fin']:.4f}]")
            avg_dist = sum(p['distance_km'] for p in selected_points) / len(selected_points)
            print(f"📏 Distância média: {avg_dist:.2f}km")
        if start_year is not None and end_year is not None:
            print(f"📅 Período: {start_year}-{end_year}")
        print(f"💾 CSV salvo em: {csv_result}")
        print("="*60)
        
        return csv_result
    except Exception as e:
        print(f"❌ Falha ao gerar CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_climbra_data(
    datasets_base_dir: str = './datasets/CLIMBRA',
    results_base_dir: str = './results/climbra',
    allowed_roots: list[str] | None = None,
    allowed_extra_files: list[str] | None = None,
    chunk_size: int = 1 << 14,
    max_retries: int = 3,
    timeout: int = 30,
    show_progress: bool = True,
) -> Path | None:
    """
    Fluxo completo: escolhe dataset → verifica existência local (espelha árvore) → baixa se necessário
    → pergunta cidade → processa NetCDF em CSV seguindo árvore de resultados e nome editável.

    - Salva datasets em `datasets/CLIMBRA/<tree>/file.nc` espelhando a árvore do servidor.
    - Salva resultados em `results/climbra/<tree>/<cidade>_<periodo>_<subset>.csv`.
    """
    # 1) Escolher dataset
    url = choose_climbra_dataset_url(
        allowed_roots=allowed_roots,
        allowed_extra_files=allowed_extra_files,
    )
    if not url:
        print('❌ Seleção cancelada ou sem URL.')
        return None

    entry = _parse_dataset_entry(url)
    segments = entry.get('segments', [])
    file_name = entry.get('file_name')
    if not file_name:
        file_name = _derive_filename_from_url(url)

    # 2) Construir caminho local espelhando árvore
    dataset_dir = Path(datasets_base_dir).joinpath(*segments)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    local_nc_path = dataset_dir / file_name

    # 3) Verificar existência e decidir ação
    if local_nc_path.exists():
        choice = questionary.select(
            "Arquivo já existe. O que deseja fazer?",
            choices=[
                "📄 Processar dataset local",
                "⬇️ Baixar novamente e sobrescrever",
                "❌ Cancelar",
            ],
        ).ask()
        if choice is None or choice.endswith('Cancelar'):
            print('❌ Operação cancelada.')
            return None
        if choice.startswith('⬇️'):
            # Overwrite
            downloaded = download_climbra_dataset(
                url=url,
                output_dir=str(dataset_dir),
                chunk_size=chunk_size,
                max_retries=max_retries,
                timeout=timeout,
                show_progress=show_progress,
                destination_path=local_nc_path,
            )
            if not downloaded:
                return None
        # else: use local file as-is
    else:
        # Não existe: sugerir download
        confirm = questionary.confirm(
            f"Dataset não encontrado localmente. Baixar para {local_nc_path}?"
        ).ask()
        if not confirm:
            print('❌ Operação cancelada pelo usuário.')
            return None
        downloaded = download_climbra_dataset(
            url=url,
            output_dir=str(dataset_dir),
            chunk_size=chunk_size,
            max_retries=max_retries,
            timeout=timeout,
            show_progress=show_progress,
            destination_path=local_nc_path,
        )
        if not downloaded:
            return None

    # 4) Cidade desejada (opcional)
    city_name = questionary.text(
        "Digite o nome da cidade (ou deixe em branco para informar coordenadas):"
    ).ask()
    if city_name is not None:
        city_name = city_name.strip()
    if not city_name:
        city_name = None

    # 5) Processar para CSV, espelhando árvore em results/climbra/<tree>
    try:
        csv_path = process_local_netcdf_with_city(
            nc_path=local_nc_path,
            city_name=city_name,
            output_dir=results_base_dir,
            results_subdir_segments=segments,
            allow_filename_edit=True,
        )
        return csv_path
    except Exception as e:
        print(f"❌ Falha no processamento do NetCDF: {e}")
        return None