## 🌐 CLIMBra — Seleção, Download e Processamento

Este projeto inclui um navegador interativo de datasets CLIMBra (Climate datasets for Brazil), com espelhamento de estrutura de pastas local e processamento automático de NetCDF → CSV. Recursos:

- 📂 Navegação por árvore com busca por nome/caminho
- 🎛️ Filtragem por pastas raiz via constantes (`AllowedRootFolders`, `AllowedExtraFiles`)
- ⬇️ Download com progresso, ETA e tentativas automáticas
- 🗂️ Espelhamento da árvore do servidor em `datasets/` e `results/`
- 🔄 Detecção de datasets já baixados para evitar downloads repetidos
- 🏙️ Seleção de cidade (geocoding) e pontos de grade mais próximos
- 🧮 Extração pontual ou média espacial, com período completo ou personalizado

### Fluxo Completo (Recomendado)

Use `get_climbra_data` para executar o fluxo ponta-a-ponta: seleção do dataset → download/uso local → seleção da cidade → geração de CSV.
v
```python
from idf_analysis.data.api import get_climbra_data, AllowedRootFolders

# Executa fluxo completo, mostrando apenas "Gridded data"
csv_path = get_climbra_data(
    datasets_base_dir='./datasets/CLIMBRA',
    results_base_dir='./results/climbra',
    allowed_roots=[AllowedRootFolders.GriddedData],
    chunk_size=1<<16,  # 64KB
    max_retries=3,
    timeout=60,
    show_progress=True,
)
print(csv_path)
```

Comportamento:
- Ao selecionar um arquivo, o sistema verifica se o NetCDF já existe em `datasets/CLIMBRA/<árvore>/<arquivo>.nc`.
- Se existir, pergunta se deseja sobrescrever ou usar o dataset local.
- Se não existir, oferece “Baixar” ou “Cancelar”.
- Após a escolha, solicita cidade ou coordenadas, encontra os pontos de grade próximos, e gera o CSV em `results/climbra/<árvore>/<nome>.csv`.

### Componentes Modulares

Se preferir separar as etapas:

```python
from idf_analysis.data.api import (
    choose_climbra_dataset_url,
    download_climbra_dataset,
    process_local_netcdf_with_city,
    AllowedRootFolders,
)

# 1) Selecionar URL
url = choose_climbra_dataset_url(
    allowed_roots=[AllowedRootFolders.GriddedData],
    allowed_extra_files=[],
)

# 2) Calcular caminho espelhado
from urllib.parse import urlparse, parse_qs
from pathlib import Path
from urllib.parse import unquote

pr = urlparse(url)
qs = parse_qs(pr.query)
path = unquote(qs.get('path', [''])[0])
file_name = unquote(qs.get('fileName', [''])[0])
segments = [s for s in path.split('/') if s and s != 'V5']
if segments and file_name and segments[-1] == file_name:
    segments = segments[:-1]  # remove pasta redundante igual ao arquivo
nc_mirrored = Path('datasets/CLIMBRA').joinpath(*segments, file_name)
nc_mirrored.parent.mkdir(parents=True, exist_ok=True)

# 3) Baixar (ou reutilizar): feito pelo fluxo completo, mas pode ser manual
download_climbra_dataset(url, destination_path=nc_mirrored)

# 4) Processar para CSV espelhado
csv_out = process_local_netcdf_with_city(
    nc_path=nc_mirrored,
    output_dir='./results/climbra',
    results_subdir_segments=segments,  # espelha árvore
    allow_filename_edit=True,
)
```

### Exemplo: Script de Download (`scripts/download_data.py`)

O projeto inclui um script interativo para baixar dados das bases CEMADEN, INMET e CLIMBra. Exemplo simplificado do uso atual:

```python
from idf_analysis.data.api.cemaden import get_cemaden_data
from idf_analysis.data.api.inmet import get_inmet_data
from idf_analysis.data.api.climbra import get_climbra_data, AllowedRootFolders

import questionary
from dotenv import load_dotenv
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(dotenv_path=os.path.join(project_root, ".env"))

def main():
  escolha = questionary.select(
    "Qual base de dados deseja baixar?",
    choices=["CEMADEN", "INMET", "CLIMBra"]
  ).ask()

  if escolha == "CEMADEN":
    get_cemaden_data()
  elif escolha == "INMET":
    get_inmet_data()
  elif escolha == "CLIMBra":
    get_climbra_data(
      allowed_roots=[AllowedRootFolders.GriddedData],
      chunk_size=1<<16,
      max_retries=3,
      timeout=60,
      show_progress=True,
    )
  else:
    print("Nenhuma opção válida selecionada. Encerrando.")

if __name__ == "__main__":
  main()
```

Para executar:

```bash
python scripts/download_data.py
```

### Filtragem por Pastas Raiz

Use `AllowedRootFolders` para controlar quais categorias aparecem:
- `AllowedRootFolders.CatchmentsDataV3` → "Catchments-Data-v3"
- `AllowedRootFolders.GriddedData` → "Gridded data"
- `AllowedRootFolders.EnsembleData` → "Ensemble data"
- `AllowedRootFolders.ETo` → "ETo"

`AllowedExtraFiles.ReadMe` inclui o arquivo "READ_ME_paper2.docx" quando desejado.

### Boas Práticas: Espelhamento de Estrutura

- Sempre mantenha a estrutura de árvore do servidor localmente em `datasets/CLIMBRA`.
- Esse espelhamento viabiliza:
  - Detecção automática de datasets já baixados
  - Organização consistente entre múltiplos modelos/cenários

Estrutura esperada (exemplo):

```
datasets/CLIMBRA/
  Gridded data/
    pr/
      hist/
        MIROC6-pr-hist.nc

results/climbra/
  Gridded data/
    pr/
      hist/
        sao_paulo_1980-2013_point.csv
```

### Regras de Nomeação de CSV

- Padrão: `{cidade_sanitizada}_{inicio}-{fim}_{subset}.csv`
- `cidade_sanitizada` remove acentos e substitui espaços por `_`.
- `subset` é `point` (1 ponto) ou `area` (múltiplos pontos selecionados).
- A edição manual do nome é permitida; a extensão `.csv` é garantida sem duplicação.

### Observações

- O download exibe progresso, velocidade média e ETA.
- O tempo disponível do dataset é detectado a partir da variável `time` do NetCDF.
- A variável padrão de precipitação é `pr` (kg m-2 s-1); é convertida para mm/dia.
- Filtros de período (`start_year`, `end_year`) são aplicados antes de salvar.