# Benchmark Jupyter Notebooks

This directory contains Jupyter notebooks for running the protein foundation model benchmark framework.

## Notebooks

### `benchmark.ipynb`

The main benchmarking notebook that demonstrates:

1. Setting up the benchmark environment
2. Loading configuration
3. Loading pretrained models
4. Processing datasets
5. Running the full benchmark pipeline
6. Visualizing and exporting results

## Usage

```bash
# Install framework
pip install -e ..

# Launch Jupyter
jupyter notebook

# Or use JupyterLab
jupyter lab
```

## Running on Kaggle

1. Install the framework in your Kaggle notebook:

```python
!git clone https://github.com/RafiOISS/protein-foundation-benchmark.git
!pip install -e protein-foundation-benchmark/

# Alternatively, install directly
!pip install git+https://github.com/RafiOISS/protein-foundation-benchmark.git
```

2. Import and use:

```python
from protein_foundation_benchmark import BenchmarkFramework
from protein_foundation_benchmark.managers import ExperimentManager, DatasetManager
from protein_foundation_benchmark.models import ESM2, ProtBERT
```

3. Follow the benchmark protocol described in `benchmark.ipynb`.

## Notes

- Notebooks are designed to be run top-to-bottom without errors
- All configuration is done through YAML files in `configs/`
- Results are automatically saved to `outputs/`
- Enable GPU acceleration in Kaggle for large models