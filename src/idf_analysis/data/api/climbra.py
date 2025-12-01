from ...helpers.climbra_drive import extract_climbra_from_drive

from importlib import resources
from geopy.geocoders import Nominatim

import questionary
import pandas as pd
import numpy as np
import time
from urllib.parse import urlparse, parse_qs, unquote
import requests
import math
from pathlib import Path


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
        return cls.all()


# Aliases para compatibilidade com código existente
ALLOWED_ROOT_FOLDERS_DEFAULT = AllowedRootFolders.default()
ALLOWED_EXTRA_FILES_DEFAULT = AllowedExtraFiles.default()

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


# ===============================
# Navegação de datasets
# ===============================

def _read_dataset_index_from_resource() -> list[str]:
    """
    Lê o arquivo de índice (TXT) com URLs de datasets do CLIMBra a partir dos recursos do pacote.

    Returns:
        list[str]: Lista de URLs (uma por linha) sem linhas vazias/comentários.
    """
    resource_name = 'climbra_datasets.txt'
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
        
    Examples:
        >>> # Usar padrão (todas as pastas)
        >>> url = choose_climbra_dataset_url()
        
        >>> # Filtrar apenas dados gridados
        >>> url = choose_climbra_dataset_url(
        ...     allowed_roots=[AllowedRootFolders.GriddedData]
        ... )
        
        >>> # Múltiplas pastas
        >>> url = choose_climbra_dataset_url(
        ...     allowed_roots=[AllowedRootFolders.GriddedData, AllowedRootFolders.ETo]
        ... )
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
            # Confirmação antes de retornar
            confirm = questionary.confirm(
                f"Baixar arquivo: {payload['name']}?"
            ).ask()
            if confirm:
                return payload['url']
            else:
                continue
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
            # Confirma
            confirm = questionary.confirm(
                f"Baixar arquivo: {sel['display']}?"
            ).ask()
            if confirm:
                return sel['url']
            else:
                continue
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
                                # \x1b[K limpa até o fim da linha para evitar artefatos como 'MB/sB/ss'
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


def choose_and_download_climbra_dataset(
    output_dir: str = './downloads/climbra',
    allowed_roots: list[str] | None = None,
    allowed_extra_files: list[str] | None = None,
    chunk_size: int = 1 << 14,
    max_retries: int = 3,
    timeout: int = 30,
    show_progress: bool = True,
) -> Path | None:
    """
    Combina seleção interativa e download em uma única chamada.

    Args:
        output_dir: Diretório onde o arquivo será salvo (default: './downloads/climbra')
        allowed_roots: Lista de pastas raiz a mostrar no navegador. Use AllowedRootFolders.* para acesso às constantes.
                      (default: AllowedRootFolders.default() - todas as pastas)
        allowed_extra_files: Lista de arquivos específicos sempre incluídos. Use AllowedExtraFiles.* para acesso.
                            (default: AllowedExtraFiles.default())
        chunk_size: Tamanho do bloco de download em bytes (default: 16384)
        max_retries: Número máximo de tentativas em caso de falha (default: 3)
        timeout: Timeout da requisição em segundos (default: 30)
        show_progress: Mostrar barra de progresso e ETA (default: True)

    Returns:
        Path | None: Caminho do arquivo salvo ou None se cancelado/falha.
        
    Examples:
        >>> # Uso básico com filtros padrão (todas as pastas)
        >>> file_path = choose_and_download_climbra_dataset()
        
        >>> # Filtrar apenas dados gridados
        >>> file_path = choose_and_download_climbra_dataset(
        ...     allowed_roots=[AllowedRootFolders.GriddedData],
        ...     output_dir='./meus_dados'
        ... )
        
        >>> # Múltiplas pastas específicas
        >>> file_path = choose_and_download_climbra_dataset(
        ...     allowed_roots=[AllowedRootFolders.GriddedData, AllowedRootFolders.ETo],
        ...     chunk_size=65536,
        ...     max_retries=5
        ... )
        
        >>> # Apenas catchments com arquivo README
        >>> file_path = choose_and_download_climbra_dataset(
        ...     allowed_roots=[AllowedRootFolders.CatchmentsDataV3],
        ...     allowed_extra_files=[AllowedExtraFiles.ReadMe],
        ...     timeout=60
        ... )
    """
    url = choose_climbra_dataset_url(
        allowed_roots=allowed_roots,
        allowed_extra_files=allowed_extra_files,
    )
    if not url:
        print('❌ Seleção cancelada ou sem URL.')
        return None
    return download_climbra_dataset(
        url=url,
        output_dir=output_dir,
        chunk_size=chunk_size,
        max_retries=max_retries,
        timeout=timeout,
        show_progress=show_progress,
    )