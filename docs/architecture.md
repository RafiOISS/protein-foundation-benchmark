# Architecture (v2)

This document describes the refactored architecture for the Protein Foundation Model Benchmark Framework.

## Layered Architecture

```
┌──────────────────────────────────────────────────┐
│                   Notebook                        │
│  (orchestration only — clone, install, import)    │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│                ProteinBenchmark                    │
│  (single public API — benchmark.run(...))          │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│                  Framework                         │
│  (experiment, pipeline, checkpoint, artifacts)     │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│                  Registries                        │
│  (ModelRegistry, DatasetRegistry, MetricRegistry)  │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│                 Interfaces                         │
│  (BaseProteinModel, BaseDataset, BaseTrainer,      │
│   BaseReporter, BaseMetric)                        │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│            Datasets / Models                       │
│  (concrete implementations)                        │
└────────────────────┬─────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────┐
│                 Trainer                            │
│  (training loop)                                   │
├───────────────────────────────────────────────────┤
│                Evaluator                           │
│  (evaluation loop)                                 │
├───────────────────────────────────────────────────┤
│                Reporter                            │
│  (result export)                                   │
└───────────────────────────────────────────────────┘
```

## Principles

1. **Notebooks contain no business logic** — they only clone, install, import, and run.
2. **Every model inherits from `BaseProteinModel`** — no direct implementations.
3. **Every dataset inherits from `BaseDataset`** — no direct implementations.
4. **Registries replace if/else chains** — models/datasets register themselves.
5. **Configuration is YAML-based with Hydra** — no argparse, no hardcoded paths.
6. **Every component is independently testable** — no hidden dependencies.
7. **Outputs never overwrite** — each experiment gets a unique directory.

## Component Map

| Layer | Location | Responsibility |
|-------|----------|----------------|
| Public API | `src/framework/benchmark.py` | `ProteinBenchmark` — single point of entry |
| Framework | `src/framework/` | Experiment lifecycle, pipeline, checkpoint, artifacts |
| Registries | `src/registry/` | ModelRegistry, DatasetRegistry, MetricRegistry |
| Interfaces | `src/interfaces/` | Abstract base classes |
| Models | `src/models/<family>/` | Concrete model implementations |
| Datasets | `src/datasets/<family>/` | Concrete dataset implementations |
| Trainer | `src/trainer/` | Training loop |
| Evaluator | `src/evaluator/` | Evaluation loop |
| Metrics | `src/metrics/` | Metric computation |
| Statistics | `src/statistics/` | Statistical tests (Wilcoxon, Friedman, Bootstrap) |
| Visualization | `src/visualization/` | Publication-quality plotting |
| Utils | `src/utils/` | I/O, logging, seed, environment |

## Adding a New Model

1. Create `src/models/<name>/` with `__init__.py` subclassing `BaseProteinModel`
2. Register: `ModelRegistry.register("name", ModelClass)`
3. Add config in `configs/models/<name>.yaml`

## Adding a New Dataset

1. Create `src/datasets/<name>/` with `__init__.py` subclassing `BaseDataset`
2. Register: `DatasetRegistry.register("name", DatasetClass)`
3. Add config in `configs/datasets/<name>.yaml`