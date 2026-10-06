# Intelligent ML Model Monitoring & Automated Retraining Platform
Group C12 | Major Project-I, Sem VII, 2026-27

## Setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
pre-commit install

## Data
python -m src.data.make_dataset
python -m src.data.split_data
