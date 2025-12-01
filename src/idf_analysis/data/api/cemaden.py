import os
import requests
import questionary
import time
import io
import zipfile
import unicodedata
import pandas as pd

from dotenv import load_dotenv
from datetime import datetime
from ..processing import aggregate_to_csv
from ..reader import process_data, DataSource
from typing import Literal, Optional

UFS_BRASIL = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO"
]

def get_token(api_url: str) -> dict:
    """
    Faz uma requisição POST para obter um token de autenticação.
    Args:
        api_url (str): A URL do endpoint de autenticação da API.
    Returns:
        dict: O JSON da resposta contendo o token.
    """
    load_dotenv()
    email = os.getenv("CEMADEN_EMAIL")
    password = os.getenv("CEMADEN_PASSWORD")

    if not email or not password:
        raise ValueError("Variáveis de ambiente API_EMAIL e API_PASSWORD não encontradas no .env")

    payload = {"email": email, "password": password}
    headers = {'Content-Type': 'application/json'}
    
    print("🔑 Autenticando e obtendo token...")
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as err:
        print(f"❌ Falha na comunicação com a API de token: {err}")
        raise

def get_cities_by_state(api_url: str, token: str, fu: str) -> list[dict]:
    """
    Busca as cidades de uma UF que possuem estações do Cemaden.
    Args:
        api_url (str): A URL do endpoint da API.
        token (str): O token de autenticação JWT.
        fu (str): A sigla da unidade federativa (UF).
    """
    if not token:
        raise ValueError("Token de autenticação não pode ser vazio.")

    # O token JWT é geralmente enviado no cabeçalho 'Authorization' como 'Bearer [token]'
    # mas seguindo a descrição, enviaremos em um header chamado 'token'.
    headers = {'token': token}
    params = {'uf': fu, 'formato': 'JSON'}

    print(f"🗺️  Buscando cidades para a UF: {fu}...")
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        cidades = response.json()
        if not cidades:
            print(f"⚠️ Nenhuma cidade com estação encontrada para {fu}.")
        return cidades
    except requests.exceptions.RequestException as err:
        print(f"❌ Falha ao buscar cidades: {err}")
        raise
    
def get_stations_by_city(api_url: str, token: str, cod_ibge: str) -> list[dict]:
    """
    Busca as estações de monitoramento de um município pelo código IBGE.
    Args:
        api_url (str): A URL do endpoint da API.
        token (str): O token de autenticação JWT.
        cod_ibge (str): O código IBGE do município.
    Returns:
        list[dict]: A lista de estações de monitoramento encontradas.
    """
    if not token:
        raise ValueError("Token de autenticação não pode ser vazio.")

    headers = {'token': token}
    params = {'codibge': cod_ibge, 'formato': 'JSON'}

    print(f"🛰️  Buscando estações para o município com código IBGE: {cod_ibge}...")
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        estacoes = response.json()
        if not estacoes:
            print(f"⚠️ Nenhuma estação encontrada para este município.")
        return estacoes
    except requests.exceptions.RequestException as err:
        print(f"❌ Falha ao buscar estações: {err}")
        raise
    
def schedule_data_request(api_url: str, token: str, params: dict) -> int | None:
    """
    Agenda a requisição de dados históricos e retorna o ID do agendamento.
    Args:
        api_url (str): A URL do endpoint de agendamento da API.
        token (str): O token de autenticação JWT.
        params (dict): Dicionário com os parâmetros necessários para o agendamento, contendo:
            - "arquivo" (str): Tipo de arquivo, geralmente "CSV".
            - "codestacao" (str): Código da estação.
            - "datafim" (str): Data final no formato esperado pela API (ex: "YYYYMMDDHHMM").
            - "datainicio" (str): Data inicial no formato esperado pela API (ex: "YYYYMMDDHHMM").
            - "rede" (str): Código da rede (ex: "11").
            - "sensor" (str): Código do sensor (ex: "10").
            - "uf" (str): Sigla da unidade federativa (UF).
    Returns:
        int | None: O ID do agendamento ou None em caso de falha.
    """
    print("\n📅 Agendando requisição de dados históricos...")
    headers = {'token': token}
    
    try:
        response = requests.get(api_url, headers=headers, params=params, timeout=15)

        response.raise_for_status()
        
        resposta_json = response.json()
        job_id = resposta_json.get("id")
        
        if job_id:
            print(f"✅ Requisição agendada com sucesso! ID do Agendamento: {job_id}")
            return job_id
        else:
            print("❌ Falha: A API não retornou um ID de agendamento.")
            print("Resposta recebida:", resposta_json)
            return None
            
    except requests.exceptions.HTTPError as http_err:
        # Adiciona um print mais detalhado para o erro 404
        if http_err.response.status_code == 404:
            print("❌ Erro 404: Endpoint não encontrado. Verifique a URL e o método (GET/POST) na documentação da API.")
        print(f"❌ Falha ao agendar requisição: {http_err}")
        print(f"Corpo da resposta do erro: {http_err.response.text}") # Ajuda a depurar
        raise
    except requests.exceptions.RequestException as err:
        print(f"❌ Falha na comunicação ao agendar requisição: {err}")
        raise

def check_scheduling_status(
    api_url: str, 
    token: str, 
    job_id: int, 
    station_name: dict, 
    selected_city: str, 
    max_attempts: int = 20, 
    delay_seconds: int = 15
) -> dict | None:
    """
    Verifica o status de um agendamento (polling) até que esteja concluído ou falhe.
    Args:
        api_url (str): A URL do endpoint de status da API.
        token (str): O token de autenticação JWT.
        job_id (int): O ID do agendamento a ser verificado.
        station_name (dict): O dicionário com os dados da estação selecionada.
        selected_city (str): O nome da cidade selecionada.
        max_attempts (int): Número máximo de tentativas de verificação.
        delay_seconds (int): Tempo em segundos entre cada tentativa.
    Returns:
        dict | None: O dicionário com os dados do agendamento final ou None se não for concluído.
    """
    print(f"\n⏳ Iniciando verificação de status para o Agendamento ID: {job_id}.")
    print("Isso pode levar alguns minutos. O script irá verificar automaticamente.")

    headers = {
        'token': token,
        'Accept': 'application/json'
    }

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"   Tentativa {attempt}/{max_attempts}... Verificando status...")
            response = requests.get(api_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            todos_agendamentos = response.json()
            # Procura o nosso agendamento específico na lista
            meu_agendamento = next((job for job in todos_agendamentos if job.get("id") == job_id), None)
            
            if meu_agendamento:
                status = meu_agendamento.get("status", {}).get("description", "DESCONHECIDO")
                print(f"   Status atual: {status}")

                if status == "CONCLUIDA":
                    print("🎉 Processamento concluído!")
                    return meu_agendamento
                elif status in ["REJEITADA", "EXPIRADA"]:
                    print(f"❌ A requisição foi '{status}'.")
                    return meu_agendamento
            
            # Se não for um status final, espera para a próxima tentativa
            time.sleep(delay_seconds)

        except requests.exceptions.RequestException as err:
            print(f"   Houve um erro na tentativa {attempt}: {err}. Tentando novamente em {delay_seconds}s.")
            time.sleep(delay_seconds)

    print(f"⌛ O tempo limite de verificação foi atingido ({max_attempts * delay_seconds}s).")
    print("\nO processamento pode estar demorando mais que o esperado.")
    print(f"Você pode verificar o status manualmente mais tarde. O ID do seu agendamento é: {job_id}")
    print(f"\nVerifique da seguinte forma:")    
    print(f"\tfinalize_request_by_id({job_id}, {station_name}, '{selected_city}')\n")

    return None

def convert_date_br_to_api(data_str_br: str, hourType: Literal["inicio", "fim"]) -> str | None:
    """
    Converte uma data do formato brasileiro (DD/MM/AAAA) para o formato da API (aaaaMMddHHmm).

    Args:
        data_str_br (str): A data no formato "DD/MM/AAAA".
        hourType (Literal["inicio", "fim"]): 'inicio' para adicionar o horário "0000" ou 'fim' para "2359".

    Returns:
        str | None: A data convertida ou None se o formato for inválido.
    """
    try:
        # 1. Tenta converter a string para um objeto de data
        data_obj = datetime.strptime(data_str_br, "%d/%m/%Y")
        
        # 2. Formata a data para 'aaaaMMdd' e adiciona o horário apropriado
        if hourType == 'inicio':
            return data_obj.strftime("%Y%m%d") + "0000"
        elif hourType == 'fim':
            return data_obj.strftime("%Y%m%d") + "2359"
        else:
            return None # Tipo inválido
            
    except ValueError:
        # Retorna None se a data digitada não corresponder ao formato "DD/MM/AAAA"
        return None

def normalize_name(name: str) -> str:
    """
    Remove acentos, converte para maiúsculas e substitui espaços por underscores.
    Exemplo: "Pluviômetro Automático A001" -> "PLUVIOMETRO_AUTOMATICO_A001"
    """
    # Normaliza para separar os caracteres dos acentos (forma NFD)
    nfkd_form = unicodedata.normalize('NFD', name)
    # Codifica para ASCII ignorando os acentos, depois decodifica de volta para string
    name_without_accents = u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

    # Converte para maiúsculas e substitui espaços
    return name_without_accents.upper().replace(' ', '_')

def generate_acronym(name: str) -> str:
    """
    Gera sigla do nome.
    Exemplo: "TAUBATE" -> "TAU"
    """
    name_without_spaces = name.replace(" ", "")
    return name_without_spaces[:3].upper()

def download_and_extract_csv(download_link: str, station_name: str, city: str) -> tuple[str, str]:
    """
    Baixa, transforma e salva o CSV. Se o arquivo já existir,
    integra os novos dados, remove duplicatas e reordena.
    Args:
        download_link (str): O link para download do arquivo .zip.
        station_name (str): O nome da estação para nomear o arquivo.
        city (str): O nome da cidade para gerar a sigla.
    Returns:
        tuple[str, str]: O caminho da pasta onde o arquivo foi salvo e o nome base formatado.
    """
    if not download_link:
        print("❌ Link de download inválido.")
        return "", ""

    print(f"\n⏬ Baixando dados do link...")
    try:
        response = requests.get(download_link)
        response.raise_for_status()
        zip_buffer = io.BytesIO(response.content)

        with zipfile.ZipFile(zip_buffer) as archive:
            for nome_arquivo_no_zip in archive.namelist():
                if nome_arquivo_no_zip.lower().endswith('.csv'):
                    print(f"📄 Arquivo CSV encontrado no .zip: {nome_arquivo_no_zip}")
                    
                    with archive.open(nome_arquivo_no_zip) as f:
                        conteudo_original_bytes = f.read()

                    # Transforma os novos dados e já os prepara em um DataFrame
                    conteudo_final_str = clean_and_transform_csv(conteudo_original_bytes)
                    df_novo = pd.read_csv(io.StringIO(conteudo_final_str), sep=';')
                    
                    # Prepara nomes da pasta e do arquivo final
                    estacao_formatada = normalize_name(station_name)
                    sigla = generate_acronym(normalize_name(city))
                    nome_base_formatado = f"{estacao_formatada}_{sigla}"
                    pasta_destino = f"./datasets/CEMADEN_{nome_base_formatado}"
                    nome_arquivo_final = f"cemaden_{nome_base_formatado.lower()}.csv"
                    os.makedirs(pasta_destino, exist_ok=True)
                    caminho_final = os.path.join(pasta_destino, nome_arquivo_final)

                    # --- LÓGICA DE INTEGRAÇÃO ---
                    # Verifica se o arquivo de destino já existe
                    if os.path.exists(caminho_final):
                        print(f"🔄 Arquivo existente encontrado. Integrando novos dados...")
                        
                        # Lê o arquivo CSV existente
                        df_existente = pd.read_csv(caminho_final, sep=';')
                        
                        # Concatena o dataframe existente com o novo
                        df_combinado = pd.concat([df_existente, df_novo], ignore_index=True)
                        
                        # Remove linhas duplicadas para garantir a integridade
                        df_combinado.drop_duplicates(inplace=True)
                        
                        # Garante que a datahora seja tratada como string para ordenação correta
                        # e depois ordena os dados para manter a consistência cronológica
                        df_combinado['datahora'] = df_combinado['datahora'].astype(str)
                        df_combinado.sort_values(by='datahora', inplace=True)
                        
                        # Salva o dataframe combinado de volta no arquivo, sobrescrevendo-o
                        df_combinado.to_csv(caminho_final, sep=';', index=False, encoding='utf-8')
                        print(f"✅ Dados integrados, duplicatas removidas e arquivo salvo em: {caminho_final}")

                    else:
                        # Se o arquivo não existe, simplesmente salva o novo dataframe
                        print(f"📝 Criando novo arquivo de dados...")
                        df_novo.to_csv(caminho_final, sep=';', index=False, encoding='utf-8')
                        print(f"✅ Arquivo criado e salvo com sucesso em: {caminho_final}")
                        
                    return pasta_destino, nome_base_formatado.lower()

            print("⚠️ Nenhum arquivo .csv foi encontrado dentro do arquivo .zip. A estação pode ter sido criada recentemente ou não ter dados disponíveis.")
            return "", ""

    except requests.exceptions.RequestException as e:
        print(f"❌ Falha ao baixar o arquivo: {e}")
    except zipfile.BadZipFile:
        print("❌ O arquivo baixado não é um .zip válido.")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado durante a extração: {e}")
    return "", ""

def clean_and_transform_csv(csv_original_content: bytes) -> str:
    """
    Lê o conteúdo de um CSV, limpa, transforma e o retorna como uma string no formato final.
    
    Args:
        csv_original_content (bytes): O conteúdo binário do arquivo CSV baixado.

    Returns:
        str: O conteúdo do novo CSV formatado como uma string.
    """
    print("🧹 Limpando e transformando os dados...")
    
    # Usa io.BytesIO para ler o conteúdo binário como se fosse um arquivo
    df = pd.read_csv(io.BytesIO(csv_original_content), sep=',')

    # 1. Manter apenas as colunas que nos interessam
    colunas_para_manter = [
        'cidade', 'codestacao', 'uf', 'nome', 
        'latitude', 'longitude', 'datahora', 'valor'
    ]
    df = df[colunas_para_manter]

    # 2. Renomear as colunas para o padrão final (camelCase)
    mapa_nomes = {
        'cidade': 'municipio',
        'codestacao': 'codEstacao',
        'nome': 'nomeEstacao',
        'valor': 'valorMedida'
    }
    df = df.rename(columns=mapa_nomes)
    
    # 3. Reordenar as colunas para a ordem final
    ordem_final = [
        'municipio', 'codEstacao', 'uf', 'nomeEstacao', 
        'latitude', 'longitude', 'datahora', 'valorMedida'
    ]
    df = df[ordem_final]
    
    # 4. Transformar os dados: trocar separador decimal de '.' para ','
    # Para fazer isso de forma segura, convertemos para string e substituímos
    for col in ['latitude', 'longitude', 'valorMedida']:
        df[col] = df[col].astype(str).str.replace('.', ',', regex=False)
        
    # Opcional: Garantir que a datahora tenha o formato com milissegundo ".0"
    df['datahora'] = df['datahora'].astype(str) + ".0"

    # 5. Gera a string final do CSV com ';' como separador e sem o índice do DataFrame
    csv_final_string = df.to_csv(sep=';', index=False)
    
    print("✨ Transformação concluída!")
    return csv_final_string

def finalize_request_by_id(
    job_id: int, 
    final_station: dict, 
    selected_city: str, 
    token: Optional[str] = None, 
    process: bool = True
) -> pd.DataFrame | None:
    """
    Verifica o status de um agendamento existente e, se concluído,
    baixa, processa e salva os dados.
    
    Args:
        job_id (int): O ID do agendamento a ser verificado.
        token (Optional[str]): O token de autenticação.
        final_station (dict): O dicionário com os dados da estação selecionada.
        selected_city (str): O nome da cidade selecionada.
        process (bool): Flag para determinar se o pós-processamento deve ocorrer.
    
    Returns:
        O DataFrame processado ou None em caso de falha.
    """
    # Define as URLs necessárias dentro da função ou passa como argumento
    TOKEN_URL = "https://sgaa.cemaden.gov.br/SGAA/rest/controle-token/tokens"
    STATUS_URL = "https://sws.cemaden.gov.br/PED/rest/controle-agendamento/agendamentos"
    
    try:
        if not token:
            resposta_token = get_token(TOKEN_URL)
            token = resposta_token.get("access_token") or resposta_token.get("token")
            if not token:
                print("❌ Não foi possível extrair o token da resposta da API.")
                return None

            print("✅ Token recebido com sucesso!\n")
    
        resultado_final = check_scheduling_status(STATUS_URL, token, job_id, final_station, selected_city)
        
        if resultado_final:
            status_final = resultado_final.get("status", {}).get("description")
            link_download = resultado_final.get("link")
            
            print("\n" + "="*50)
            print("🏁 Processo Finalizado!")
            print(f"Status Final da Requisição: {status_final}")
            
            if status_final == "CONCLUIDA" and link_download:
                print("✅ Sucesso! O link para download do seu arquivo é:")
                print(f"\n   ➡️   {link_download}\n")
                
                # Nota: Ajustei a chamada para a sua versão com 3 argumentos
                # Verifique se a sua função `download_and_extract_csv` realmente precisa de `selected_city`
                
                nome_da_estacao = final_station.get('nome', 'ESTACAO_DESCONHECIDA')
                
                caminho_final, nome_limpo = download_and_extract_csv(link_download, nome_da_estacao, selected_city)
                
                # Pós‑processamento
                if process and caminho_final:
                    print("\nIniciando pós-processamento dos dados...")
                    df = process_data(
                        source=DataSource.CEMADEN,
                        data_path=caminho_final,
                        site_filter="API"
                    )
                    
                    aggregate_to_csv(
                        df=df,
                        name='cemaden_' + nome_limpo,
                        directory='./results/cemaden_' + nome_limpo,
                        #include_minutes=True,
                        #minute_freq=5
                    )
                    
                    return df
                elif caminho_final:
                    # Se o processamento não for necessário, podemos retornar o caminho do arquivo salvo
                    print("Pós-processamento não solicitado. Arquivo bruto salvo.")
                    return None
                    
            else:
                print("❌ Não foi possível obter o link de download.")
                print("="*50)
        return None
    
    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"\nOcorreu um erro e o programa será encerrado: {e}")
    except (KeyboardInterrupt, SystemExit) as e:
        if str(e): print(f"\n👋 {e}")
        else: print("\n👋 Operação interrompida pelo usuário.")
    return None

def get_cemaden_data():
    """
    Executa o fluxo completo e interativo de coleta e processamento de dados do Cemaden.
    """
    # 1. Defina as URLs dos seus webservices
    TOKEN_URL = "https://sgaa.cemaden.gov.br/SGAA/rest/controle-token/tokens"
    CIDADES_URL = "https://sws.cemaden.gov.br/PED/rest/pcds-cadastro/cidades"
    ESTACOES_URL = "https://sws.cemaden.gov.br/PED/rest/pcds-cadastro/estacoes"
    AGENDAMENTO_URL = "https://sws.cemaden.gov.br/PED/rest/controle-agendamento/pcds-dados-historicos"

    try:
        # Etapa 1: Obter o token
        resposta_token = get_token(TOKEN_URL)
        meu_token = resposta_token.get("access_token") or resposta_token.get("token")
        if not meu_token:
            print("❌ Não foi possível extrair o token da resposta da API.")
            return

        print("✅ Token recebido com sucesso!\n")

        # Etapa 2: Escolher UF
        uf_escolhida = questionary.select("Selecione um estado (UF):", choices=sorted(UFS_BRASIL)).ask()
        if uf_escolhida is None: 
            print("Operação cancelada.")
            return

        # Etapa 3: Escolher Cidade
        lista_cidades = get_cities_by_state(CIDADES_URL, meu_token, uf_escolhida)
        if not lista_cidades: return
        
        nomes_cidades = [c["cidade"] for c in lista_cidades]
        cidade_escolhida = questionary.select(f"Selecione a cidade em {uf_escolhida}:", choices=sorted(nomes_cidades)).ask()
        if cidade_escolhida is None:
            print("Operação cancelada.")
            return

        cod_ibge_selecionado = next((c["codibge"] for c in lista_cidades if c["cidade"] == cidade_escolhida), None)
        if not cod_ibge_selecionado:
            print(f"❌ Erro: não foi possível encontrar o código IBGE para {cidade_escolhida}.")
            return
        
        # Etapa 4: Buscar e escolher a estação
        lista_estacoes = get_stations_by_city(ESTACOES_URL, meu_token, str(cod_ibge_selecionado))
        if not lista_estacoes: return
        
        opcoes_estacao = [f"{e.get('codestacao', 'S/C')} - {e.get('nome', 'S/N')}" for e in lista_estacoes]
        estacao_selecionada_str = questionary.select(f"Selecione a estação em {cidade_escolhida}:", choices=opcoes_estacao).ask()
        if estacao_selecionada_str is None:
            print("Operação cancelada.")
            return
        
        estacao_final = next((e for e in lista_estacoes if f"{e.get('codestacao', 'S/C')} - {e.get('nome', 'S/N')}" == estacao_selecionada_str), None)

        # Etapa 5: Coletar datas e agendar
        print("\n--- Agendamento de Dados Históricos ---")
        print("Lembre-se: dados do Cemaden são mais consistentes a partir de 2013.")
        
        # (O bloco de coleta e validação de datas permanece o mesmo aqui...)
        while True:
            data_inicio_br = questionary.text("Digite a data inicial (formato DD/MM/AAAA):").ask()
            if data_inicio_br is None: print("Operação cancelada."); return
            if convert_date_br_to_api(data_inicio_br, 'inicio'): break 
            print("❌ Formato de data inválido. Por favor, use DD/MM/AAAA.")

        data_inicio_obj = datetime.strptime(data_inicio_br, "%d/%m/%Y")

        print("\n" + "-"*60)
        print("ℹ️ INFORMAÇÕES IMPORTANTES SOBRE O PERÍODO DE CONSULTA:")
        print(f"   - A data final não pode ser anterior a {data_inicio_br}.")
        print(f"   - Se precisar de um período maior, você pode rodar o script")
        print(f"     novamente para a mesma estação. Os dados serão integrados automaticamente.")
        print("-" * 60, "\n")

        while True:
            data_fim_br = questionary.text("Digite a data final (formato DD/MM/AAAA):").ask()
            if data_fim_br is None: print("Operação cancelada."); return
            data_fim_api = convert_date_br_to_api(data_fim_br, 'fim')
            if not data_fim_api: print("❌ Formato de data inválido. Por favor, use DD/MM/AAAA."); continue
            data_fim_obj = datetime.strptime(data_fim_br, "%d/%m/%Y")
            if data_fim_obj < data_inicio_obj: print(f"❌ Erro: A data final ({data_fim_br}) não pode ser anterior à data inicial ({data_inicio_br})."); continue
            break

        data_inicio_api = convert_date_br_to_api(data_inicio_br, 'inicio')
        params_agendamento = {
            "arquivo": "CSV", 
            "codestacao": estacao_final.get('codestacao'),
            "datafim": data_fim_api, 
            "datainicio": data_inicio_api, 
            "rede": "11", 
            "sensor": "10",
            "uf": uf_escolhida,
        }

        job_id = schedule_data_request(AGENDAMENTO_URL, meu_token, params_agendamento)

        if job_id:
            # Agora a main apenas delega a finalização para a nova função
            finalize_request_by_id(job_id, estacao_final, cidade_escolhida, meu_token, process=True)

    except (ValueError, requests.exceptions.RequestException) as e:
        print(f"\nOcorreu um erro e o programa será encerrado: {e}")
    except (KeyboardInterrupt, SystemExit) as e:
        if str(e): print(f"\n👋 {e}")
        else: print("\n👋 Operação interrompida pelo usuário.")