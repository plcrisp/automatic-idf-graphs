## 🌐 CLIMBra Dataset Selection & Download

The project includes an interactive terminal navigator for browsing and downloading CLIMBra (Climate datasets for Brazil) gridded datasets. This feature allows you to:

- 📂 Browse the CLIMBra dataset index through an intuitive folder tree
- 🔍 Search datasets by name or path
- ⬇️ Download with progress tracking and automatic retriesv
- 🎯 Filter datasets by root folders for easier navigation

### Basic Usage

```python
from idf_analysis.data.api import choose_and_download_climbra_dataset

# Interactive selection and download with default filters
# (Shows: Catchments-Data-v3, Gridded data, Ensemble data, ETo, READ_ME_paper2.docx)
file_path = choose_and_download_climbra_dataset(
    output_dir='./downloads/climbra'
)
```

### Custom Root Filtering

You can customize which root folders are shown in the navigator using the `AllowedRootFolders` and `AllowedExtraFiles` classes:

```python
from idf_analysis.data.api import (
    choose_and_download_climbra_dataset,
    AllowedRootFolders,
    AllowedExtraFiles
)

# Show only specific categories using class attributes
file_path = choose_and_download_climbra_dataset(
    allowed_roots=[AllowedRootFolders.GriddedData, AllowedRootFolders.ETo],
    allowed_extra_files=[AllowedExtraFiles.ReadMe],
    output_dir='./downloads/climbra'
)
```

### Advanced: Separate Selection and Download

```python
from idf_analysis.data.api import (
    choose_climbra_dataset_url,
    download_climbra_dataset,
    AllowedRootFolders
)

# Step 1: Browse and select dataset
url = choose_climbra_dataset_url(
    allowed_roots=[AllowedRootFolders.GriddedData],
    allowed_extra_files=[]
)

if url:
    # Step 2: Download with custom settings
    file_path = download_climbra_dataset(
        url=url,
        output_dir='./my_datasets',
        chunk_size=65536,  # 64KB chunks
        max_retries=5,
        show_progress=True  # Shows progress bar with ETA
    )
```

### Available Root Folders

Use the `AllowedRootFolders` class to specify which categories to show:

- `AllowedRootFolders.CatchmentsDataV3` - Catchment-specific precipitation data (`"Catchments-Data-v3"`)
- `AllowedRootFolders.GriddedData` - Spatially gridded climate datasets (`"Gridded data"`)
- `AllowedRootFolders.EnsembleData` - Climate model ensemble projections (`"Ensemble data"`)
- `AllowedRootFolders.ETo` - Evapotranspiration reference data (`"ETo"`)

Use the `AllowedExtraFiles` class for specific files:

- `AllowedExtraFiles.ReadMe` - Documentation file (`"READ_ME_paper2.docx"`)

**Helper methods:**
- `AllowedRootFolders.all()` - Returns all available folders
- `AllowedRootFolders.default()` - Returns default selection