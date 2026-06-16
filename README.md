# ☔ Automatic IDF Graphs

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9+-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/github/license/plcrisp/Automatic-IDF-Graphs?color=green">
  <img alt="Made with Jupyter" src="https://img.shields.io/badge/made%20with-Jupyter-F37626?logo=jupyter&logoColor=white">
</p>

A Python library for end-to-end **Intensity-Duration-Frequency (IDF)** curve analysis — from raw historical rainfall records to bias-corrected future climate scenarios.

## 🌧️ About

This project was born out of an undergraduate scientific research initiative in hydrology, **funded by FAPEMIG** and supervised at UNIFEI. It automates the full IDF analysis pipeline that researchers and engineers normally do by hand: collecting rainfall data straight from official sources, treating and validating the series, disaggregating and modeling extremes, and generating IDF curves for both historical records and future climate projections.

## ✨ Features

- 🛰️ **Automatic data acquisition** from INMET and CEMADEN
- 🧹 **Data treatment** — gap filling using an auxiliary station via Random Forest, station correlation via the Pearson coefficient and double-mass curves, and Mann-Kendall trend tests
- 🎲 **Rainfall disaggregation** using CETESB factors or the Bartlett-Lewis stochastic model
- 📈 **Extreme value modeling** with statistical distribution fitting
- ☔ **Historical IDF curves**
- 🌍 **Future IDF curves** combining CLIMBra climate projections with bias correction
- 🗺️ **Visualization-ready** — interactive maps and ready-to-export plots

## 📓 Notebooks

The fastest way to see what this library actually does is to walk through the notebooks — each one explores a technique used in the pipeline above.

| Notebook | What it covers |
|---|---|
| `01_Data_Acquisition_and_Preparation` | Downloads and structures rainfall series from INMET/CEMADEN |
| `02_Quality_Analysis_and_Exploratory_Statistics` | Data treatment: gap filling with an auxiliary station (Random Forest), consistency checks via Pearson correlation and double-mass curves, and trend tests |
| `03_Historical_IDF_Curve_Generation` | Extreme value modeling and historical IDF curve generation |
| `04_Future_IDF_Curve_Generation_with_Bias_Correction` | Future IDF curves combining CLIMBra projections with bias correction |
| `05_Bartlett_Lewis_Disaggregation` | Rainfall disaggregation into sub-daily intervals using the Bartlett-Lewis stochastic model |
| `Usage_Flow` | End-to-end example tying the whole pipeline together |

> Notebooks 01–05 explore each technique in depth. Disaggregation comes before extreme value modeling in the actual pipeline (diagram above) because the IDF curve needs sub-daily duration series, which only exist once CETESB factors or the Bartlett-Lewis model break the daily data into finer intervals.

👉 New here? Start with **`Usage_Flow.ipynb`**.

## ⚙️ Installation

```bash
git clone https://github.com/plcrisp/Automatic-IDF-Graphs.git
cd Automatic-IDF-Graphs
pip install -e .
```

Requires **Python 3.9+**. Some integrations (like CLIMBra access via Google Drive) need API credentials — set them in a local `.env` file.

## 🗂️ Project Structure

```
Automatic-IDF-Graphs/
├── notebooks/              # Step-by-step pipeline notebooks
├── scripts/                # Standalone helper scripts (e.g., download_data.py)
└── src/idf_analysis/
    ├── analysis/
    │   ├── historical/     # IDF curves, intervals, sub-daily extremes, trend & consistency tests, Bartlett-Lewis
    │   └── projection/     # Future scenarios & bias correction (quantile mapping, EQM, DBC)
    ├── core/                # Statistical distributions & correlation
    ├── data/                # Readers & processing utilities
    │   └── api/             # INMET, CEMADEN and CLIMBra data connectors
    ├── helpers/             # Notebook & Google Drive helpers
    ├── resources/           # Reference datasets (station lists, disaggregation factors, CABra catchments)
    └── visualization/       # Plotting utilities
```

## 🌐 Data Sources

- **INMET** — Instituto Nacional de Meteorologia — https://www.inmet.gov.br
- **CEMADEN** — Centro Nacional de Monitoramento e Alertas de Desastres Naturais — http://www.cemaden.gov.br
- **CLIMBra** — bias-corrected CMIP6 climate projections for Brazil (Ballarin et al., 2023, *Scientific Data*) — https://doi.org/10.1038/s41597-023-01956-z

## 📜 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 📚 Citation

This code was developed as part of the undergraduate research project "Development of a Python Library for Rainfall Data Processing and Intensity-Duration-Frequency (IDF) Curve Construction", funded by FAPEMIG. The project was supervised by Prof. Marina Batalini de Macedo and carried out by Pedro Lucas Crisp Pinto and Bruno Peres dos Santos.

If you use this code in your research, publications, or projects, please cite it as follows:

> PINTO, P. L. C.; DOS SANTOS, B. P.; MACEDO, M. B. Precipitation Analysis: Automatic IDF Graphs. Available at: https://github.com/plcrisp/Automatic-IDF-Graphs

## 👥 Authors

- **Pedro Lucas Crisp Pinto** — pedrolcrisp@gmail.com
- **Bruno Peres dos Santos**
- Supervised by **Prof. Marina Batalini de Macedo**