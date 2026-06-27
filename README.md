# Protein Foundation Model Benchmark Framework

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/release/python-312/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.30+-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A production-quality benchmark framework for comparing pretrained protein foundation models across multiple datasets and tasks. Designed for reproducibility, extensibility, and publication-quality results.

## Features

- **Multiple Foundation Models**: ESM-2, ProtBERT, ProtT5, ProteinBERT, plus CNN/BiLSTM baselines
- **Multiple Benchmark Tasks**: Fluorescence, Stability, PPI, Secondary Structure, Remote Homology, Localization, GO Prediction
- **Automatic Checkpointing**: Resume training, track best models
- **Artifact Tracking**: Version datasets, embeddings, predictions, figures
- **Statistical Rigor**: Wilcoxon, bootstrap, Friedman tests with multiple comparison correction
- **Publication-Quality Figures**: Matplotlib/Seaborn with multiple export formats
- **Reproducibility**: Deterministic seeds, environment capture, git tracking
- **Modular Design**: Independent managers for checkpoints, artifacts, datasets, pipelines, exports, experiments

## Installation

### Using pip

```bash
git clone https://github.com/RafiOISS/protein-foundation-benchmark.git
cd protein-foundation-benchmark
pip install -e .
```

### Using conda

```bash
git clone https://github.com/RafiOISS/protein-foundation-benchmark.git
cd protein-foundation-benchmark
conda env create -f environment.yml
conda activate protein-benchmark
```

### Dependencies

**Core:**
- Python 3.12+
- PyTorch 2.0+
- HuggingFace Transformers 4.30+
- NumPy, pandas, scikit-learn
- Matplotlib, Seaborn
- PyYAML, rich

**Optional (for ProteinBERT):**
- TensorFlow 2.13+
- proteinbert package

## Quick Start

```python
from src.managers import ExperimentManager, DatasetManager
from src.models import ESM2, ProtBERT
from src.managers.pipeline_manager import PipelineManager, PipelineConfig

# Create experiment manager
exp_manager = ExperimentManager("outputs/experiments")

# Create experiment
exp = exp_manager.create_experiment(
    name="esm2_vs_protbert_fluorescence",
    models=["esm2_t6_8M_UR50D", "protbert"],
    datasets=["fluorescence"],
    task_types=["regression"]
)

# Run benchmark
def run_pipeline(pipeline_manager):
    # Load data
    dataset_manager = pipeline_manager.dataset_manager
    train_loader, val_loader, test_loader = dataset_manager.get_train_val_test_loaders(
        "fluorescence"
    )

    # Load model
    model = ESM2.from_pretrained("facebook/esm2_t6_8M_UR50D")
    pipeline_manager.load_model(model)

    # Run pipeline
    return pipeline_manager.run_pipeline(train_loader, val_loader, test_loader)

results = exp_manager.run_benchmark(
    exp.experiment_id,
    [("esm2_t6_8M_UR50D", "fluorescence", "regression"),
     ("protbert", "fluorescence", "regression")],
    run_pipeline
)
```

## Repository Structure

```
protein-foundation-benchmark/
├── configs/                 # YAML configuration files
│   ├── config.yaml         # Main framework config
│   ├── models.yaml         # Model configurations
│   ├── datasets.yaml       # Dataset configurations
│   └── training.yaml       # Training hyperparameters
├── metadata/               # Framework metadata
│   ├── manifest.json       # Project manifest
│   ├── pipeline.json       # Pipeline definition
│   └── environment.json    # Environment specification
├── src/                    # Source code
│   ├── managers/           # Core managers
│   ├── models/             # Model wrappers
│   ├── datasets/           # Dataset classes
│   ├── evaluation/         # Metrics & evaluation
│   ├── plotting/           # Publication figures
│   └── utils/              # Utilities (I/O, logging, seeds)
├── outputs/                # Generated outputs (gitignored)
│   ├── checkpoints/
│   ├── embeddings/
│   ├── predictions/
│   ├── metrics/
│   ├── figures/
│   └── exports/
├── docs/                   # Documentation
├── notebooks/              # Example notebooks
└── tests/                  # Unit tests
```

## Supported Models

| Model Family | Variants | Parameters | Source |
|-------------|----------|------------|--------|
| ESM-2 | 8M, 35M, 150M, 650M, 3B, 15B | 8M - 15B | Facebook/Meta |
| ProtBERT | BFD, UniRef100 | 420M | Rostlab |
| ProtT5 | XL-UniRef50, XL-BFD | 3B | Rostlab |
| ProteinBERT | - | 110M | Custom (TF) |
| CNN Baseline | - | ~2M | This framework |
| BiLSTM Baseline | - | ~5M | This framework |

## Supported Tasks & Datasets

| Task | Dataset | Type | Metrics |
|------|---------|------|---------|
| Fluorescence Prediction | TAPE/FLIP | Regression | Spearman, Pearson, MSE, MAE |
| Stability Prediction | TAPE/FLIP | Regression | Spearman, Pearson, MSE, MAE |
| Protein-Protein Interaction | TAPE/FLIP | Binary Classification | Accuracy, F1, AUROC, AUPRC, MCC |
| Secondary Structure (SS3/SS8) | CASP/TS115 | Token Classification | Q3/Q8 Accuracy, F1 |
| Remote Homology | SCOP | Multiclass Classification | Accuracy, F1 (fold/superfamily/family) |
| Subcellular Localization | DeepLoc | Multilabel Classification | F1, AUROC, Hamming Loss |
| Gene Ontology | CAFA | Multilabel Classification | Fmax, Smin, AUROC |

## Configuration

All configuration is done via YAML files in `configs/`:

- `config.yaml`: Framework settings (paths, logging, device, checkpointing)
- `models.yaml`: Model definitions and benchmark selections
- `datasets.yaml`: Dataset configurations and metrics
- `training.yaml`: Training hyperparameters per task type

## Outputs

All outputs are organized in `outputs/`:

```
outputs/
├── checkpoints/     # Model checkpoints (best, last, periodic)
├── datasets/        # Processed datasets & cache
├── embeddings/      # Extracted embeddings (.pt, .npz)
├── experiments/     # Per-experiment results
├── exports/         # Exported results (CSV, JSON, Parquet)
├── figures/         # Publication figures (PNG, PDF)
├── logs/            # Training/evaluation logs
├── metrics/         # Computed metrics
├── models/          # Exported models (ONNX, TorchScript)
└── predictions/     # Model predictions
```

## Reproducibility

The framework ensures reproducibility through:

- Fixed random seeds (default: 42)
- Deterministic PyTorch algorithms
- Environment capture (versions, GPU info, git commit)
- Artifact versioning with checksums
- Complete configuration serialization

## Extending the Framework

### Adding a New Model

1. Create wrapper in `src/models/` inheriting from `BaseProteinModel`
2. Add configuration to `configs/models.yaml`
3. Register in `src/models/__init__.py`

### Adding a New Dataset

1. Create dataset class in `src/datasets/` inheriting from `BaseDataset`
2. Add configuration to `configs/datasets.yaml`
3. Register with `DatasetManager`

### Adding a New Metric

1. Add function to `src/evaluation/metrics.py`
2. Include in `get_default_metrics()` for task type

## Documentation

- [Architecture](docs/architecture.md) - System design and component interactions
- [Benchmark Protocol](docs/benchmark_protocol.md) - Standardized evaluation procedure
- [Reproducibility](docs/reproducibility.md) - Ensuring reproducible results
- [Roadmap](docs/roadmap.md) - Future development plans

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{protein_foundation_benchmark,
  title = {Protein Foundation Model Benchmark Framework},
  author = {RafiOISS},
  year = {2024},
  url = {https://github.com/RafiOISS/protein-foundation-benchmark}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

## Contact

- GitHub: [@RafiOISS](https://github.com/RafiOISS)
- Issues: [GitHub Issues](https://github.com/RafiOISS/protein-foundation-benchmark/issues)