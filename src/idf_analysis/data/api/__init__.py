"""API módulos para obtenção de dados meteorológicos.

Camada pública consolidada para seleção, download e processamento de datasets
CLIMBra, incluindo detecção de arquivos já baixados com estrutura espelhada.
"""

from .climbra import (
    # Fluxo completo (seleção → download/uso-local → processamento CSV)
    get_climbra_data,
    # Componentes modulares
    choose_climbra_dataset_url,
    download_climbra_dataset,
    netcdf_precipitation_to_csv,
    process_local_netcdf_with_city,
    # Constantes de filtro para o navegador
    AllowedRootFolders,
    AllowedExtraFiles,
)

__all__ = [
    'get_climbra_data',
    'choose_climbra_dataset_url',
    'download_climbra_dataset',
    'netcdf_precipitation_to_csv',
    'process_local_netcdf_with_city',
    'AllowedRootFolders',
    'AllowedExtraFiles',
]
