# Gijón pig iron production prediction

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Predicting pig iron production using Sentinel-2 satellite thermal imagery with Gaussian Process Regression (GPR).

## Overview

This project uses thermal signatures from satellite data to estimate monthly pig iron production at the Gijón steel mill. The approach leverages cloud-aware feature extraction and GPR for uncertainty-aware predictions.

**📊 For detailed analysis and methodology, see the [Analytical Report](docs/analytical_report.md).**

## Key results

- **Performance**: R² = 0.338 (LOO-CV), RMSE = 43.5k tons
- **Uncertainty**: Well-calibrated 95% prediction intervals (95.8% coverage)
- **Data**: 24 months of production data with 290 satellite observations

## Quickstart

### Installation

```bash
git clone https://github.com/mbkers/gijon-pig-iron-prediction.git
cd gijon-pig-iron-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Usage

```bash
python -m src.main --data-path "path/to/Gijon/data"
```

### Notebooks

Explore the analysis through interactive notebooks:
- **[00_quickstart.ipynb](notebooks/00_quickstart.ipynb)** - Quick start guide
- **[01_eda.ipynb](notebooks/01_eda.ipynb)** - Exploratory data analysis
- **[02_model_results.ipynb](notebooks/02_model_results.ipynb)** - Results analysis

## Project structure

```
gijon-pig-iron-prediction
├── docs/analytical_report.md       # Analysis report
├── notebooks/                      # Interactive analysis notebooks
├── src/                            # Core implementation
│   ├── data_loader.py              # Data loading utilities
│   ├── feature_engineering.py      # Feature extraction
│   ├── model.py                    # GPR implementation
│   ├── utils.py                    # Visualisation utilities
│   └── main.py                     # Main pipeline
├── results/                        # Output directory
└── tests/                          # Basic tests
```

## Author

Maximilian B. R. Kerslake