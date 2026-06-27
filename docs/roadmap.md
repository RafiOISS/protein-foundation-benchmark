# Roadmap

## Version 0.1.0 - Framework Foundation (Current)

- [x] Project structure and configuration
- [x] Core manager classes (Checkpoint, Artifact, Dataset, Pipeline, Export, Experiment)
- [x] Model wrappers (ESM-2, ProtBERT, ProtT5, ProteinBERT, CNN, BiLSTM)
- [x] Dataset base classes (BaseDataset, TabularDataset, InMemoryDataset)
- [x] Evaluation metrics and evaluator
- [x] Statistical tests (Wilcoxon, t-test, Bootstrap, Friedman, Nemenyi)
- [x] Publication-quality plotting
- [x] I/O, logging, seeding, environment utilities
- [x] Configuration system (YAML-based)
- [x] Metadata and manifest tracking
- [x] Documentation (architecture, protocol, reproducibility, roadmap)

## Version 0.2.0 - Dataset & Model Expansion

- [ ] Add complete dataset implementations (Fluorescence, Stability, PPI, SS, Homology, Localization)
- [ ] Add dataset downloading scripts
- [ ] Add more model variants (ESM-2 15B, Ankh, ProtGPT2)
- [ ] Add LoRA and adapter fine-tuning support
- [ ] Add gradient checkpointing for large models
- [ ] Add mixed precision training
- [ ] Implement cross-validation framework

## Version 0.3.0 - Pipeline & Results

- [ ] Automated benchmark runner script
- [ ] Result aggregation and comparison tables
- [ ] Benchmark report generation (HTML/PDF)
- [ ] Interactive leaderboard (streamlit/plotly-dash)
- [ ] Batch processing for multiple model-dataset pairs
- [ ] Experiment comparison dashboard
- [ ] Performance profiling and memory tracking

## Version 1.0.0 - Production Release

- [ ] Complete test suite (unit + integration)
- [ ] GitHub Actions CI/CD pipeline
- [ ] Automated benchmarking workflow
- [ ] Pre-computed benchmark results
- [ ] PyPI package release
- [ ] Full API documentation (readthedocs)
- [ ] Example notebooks for all supported tasks

## Future Considerations

### Model Support
- [ ] Ankh (Elnaggar et al.)
- [ ] ProtGPT2 (Trinidad et al.)
- [ ] MSA Transformer
- [ ] OmegaFold
- [ ] AlphaFold embedding extraction
- [ ] ESM-3 (when available)

### Dataset Support
- [ ] Mutation stability (ProTherm, MegaScale)
- [ ] Binding affinity (PDBbind)
- [ ] Enzyme classification (BRENDA)
- [ ] Protein-protein docking
- [ ] Protein-ligand interaction
- [ ] Protein design tasks
- [ ] Clinical variant prediction

### Framework Features
- [ ] Distributed training (DDP, DeepSpeed, FSDP)
- [ ] Hyperparameter optimization (Optuna, Ray Tune)
- [ ] Model ensembling
- [ ] Active learning for benchmark design
- [ ] Interpretability (attention visualization, feature attribution)
- [ ] Cloud integration (S3, GCS, Azure)
- [ ] MLflow/WandB integration
- [ ] Docker/Singularity containers
- [ ] Benchmark result database (SQLite/PostgreSQL)
- [ ] REST API for benchmark queries

### Community
- [ ] Contribution guidelines
- [ ] Code of conduct
- [ ] Issue templates
- [ ] Pull request template
- [ ] Community benchmark submissions
- [ ] Weekly/monthly benchmark updates