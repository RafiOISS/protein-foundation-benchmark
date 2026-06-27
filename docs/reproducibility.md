# Reproducibility

## Ensuring Reproducible Results

The framework is designed for full reproducibility of all benchmark experiments. This document describes the measures taken and how to use them.

## Mechanisms

### Deterministic Algorithms

The framework uses several mechanisms to ensure deterministic execution:

```python
from src.utils.seed import set_seed

# Set all seeds deterministically
set_seed(seed=42, deterministic=True)
```

This sets:
- Python random seed
- NumPy random seed
- PyTorch manual seed (CPU, CUDA, MPS)
- CUDA deterministic algorithms
- Environment hash seed
- DataLoader worker seeds

### Environment Capture

Every experiment captures the execution environment:

```python
from src.utils.environment import log_environment_info, get_git_info

# Capture environment
env_info = log_environment_info()
git_info = get_git_info()
```

Captured information:
- Python version
- PyTorch version
- CUDA/cuDNN versions
- GPU count and type
- Operating system
- Git commit hash
- Git branch
- Uncommitted changes flag

### Configuration Serialization

Every experiment saves its complete configuration:

- Framework config (`config.yaml`)
- Model config (`models.yaml`)
- Dataset config (`datasets.yaml`)
- Training config (`training.yaml`)
- Experiment-specific overrides

### Artifact Versioning

All artifacts are tracked with:

- SHA256 checksums
- Size metadata
- Creation timestamps
- Version strings
- Parent artifact relationships

```python
from src.managers.artifact_manager import ArtifactManager

manager = ArtifactManager(
    artifact_dir="outputs/experiments",
    experiment_id="exp_001",
    run_id="run_001"
)

# Track artifacts with checksums
manager.track_embeddings(
    name="esm2_fluorescence_embeddings",
    embeddings_path="outputs/embeddings/esm2_fluorescence.pt",
    metadata={"shape": [500, 320], "pooling": "mean"}
)
```

### Checkpoint Resume

Training can be resumed from checkpoints:

```python
from src.managers.checkpoint_manager import CheckpointManager

checkpoint_mgr = CheckpointManager("outputs/checkpoints/exp_001")

# Resume from best checkpoint
checkpoint = checkpoint_mgr.load(
    model=model,
    optimizer=optimizer,
    load_best=True
)
```

## Best Practices

### Before Running

1. Set `PYTHONHASHSEED` before starting Python (handled automatically by `set_seed`)
2. Ensure all dependencies are pinned to specific versions (use `environment.yml`)
3. Record the platform and hardware configuration
4. Initialize Git repository (handled at project setup)

### During Execution

1. Always call `set_seed()` before any randomized operations
2. Use `seed_worker` in DataLoader
3. Log configuration at experiment start
4. Track all created artifacts

### After Execution

1. Save experiment summary with environment info
2. Export manifest with artifact checksums
3. Commit configuration files to Git
4. Tag the experiment commit

## Verifying Reproducibility

```python
from src.utils.seed import set_seed
from src.utils.environment import log_environment_info

# Run experiment 1
set_seed(42)
result1 = run_benchmark()

# Run experiment 2 (exact same setup)
set_seed(42)
result2 = run_benchmark()

# Verify equivalence
assert result1["metrics"] == result2["metrics"]
```

## Known Non-Deterministic Operations

Even with all seeds fixed, some operations may produce non-deterministic results:

1. **PyTorch CUDA convolution ops**: Some algorithms are inherently non-deterministic on GPU
2. **CuDNN autotuning**: Disabled when `deterministic=True`
3. **Multi-GPU training**: Sequence of operations may vary
4. **Atomic operations**: GPU operations like `scatter_add` may be non-deterministic

For fully deterministic results:
- Use CPU (significant performance impact)
- Set `torch.use_deterministic_algorithms(True)`
- Avoid operations with known non-deterministic implementations