# Benchmark Protocol

## Standardized Evaluation Procedure

This document defines the standard protocol for evaluating protein foundation models within this framework.

## Experimental Design

### Model Selection

1. **Foundation Models**: Choose models from `configs/models.yaml`. Default set:
   - ESM-2 (8M, 35M, 150M, 650M, 3B, 15B)
   - ProtBERT
   - ProtT5-XL
   - ProteinBERT (if TensorFlow available)

2. **Baselines**: Include simple baselines for comparison:
   - CNN Baseline
   - BiLSTM Baseline
   - Random / Mean baseline

### Dataset Selection

Choose datasets from `configs/datasets.yaml`. Default benchmark set:
- Fluorescence Prediction (regression)
- Stability Prediction (regression)
- Protein-Protein Interaction (binary classification)
- Secondary Structure SS3 (token classification)
- Remote Homology Detection (multiclass classification)
- Subcellular Localization (multilabel classification)

### Task Types and Metrics

| Task Type | Primary Metric | Secondary Metrics |
|-----------|---------------|-------------------|
| Regression | Spearman Correlation | Pearson, MSE, MAE, R2 |
| Binary Classification | AUROC | F1, Accuracy, Precision, Recall, MCC, AUPRC |
| Multiclass Classification | F1 Macro | Accuracy, F1 Micro, F1 Weighted |
| Multilabel Classification | F1 Macro | F1 Micro, AUROC Macro, AUROC Micro, Hamming Loss |
| Token Classification | Q3/Q8 Accuracy | F1 Macro, F1 Micro |

## Standard Training Protocol

### Data Processing

1. Sequences are tokenized using model-specific tokenizers
2. Maximum sequence length: 1022 tokens (with special tokens)
3. Sequences exceeding max length are truncated from the right
4. Training/validation/test splits are preserved as provided

### Fine-tuning

- Epochs: 10
- Optimizer: AdamW
- Learning rate: 1e-4 with cosine warmup schedule
- Warmup steps: 500
- Batch size: 32
- Gradient clipping: 1.0
- Early stopping: patience 10, monitor val_loss
- Freezing: begin with frozen base, gradually unfreeze

### Embedding Extraction

- Layers: All (returns mean of all layer outputs)
- Pooling: Mean
- Batch size: 32
- Format: PyTorch tensors (.pt) and NumPy (.npz)

## Evaluation Protocol

1. **Validation during training**: Monitor all metrics on validation set after each epoch
2. **Final evaluation**: Run on held-out test set using best checkpoint
3. **Statistical significance**: Report Wilcoxon signed-rank test and bootstrap confidence intervals

### Cross-Validation

For small datasets, use 5-fold stratified cross-validation. Report mean and standard deviation across folds.

## Statistical Analysis

### Pairwise Comparison

For each pair of models (baseline vs. candidate):
1. Compute paired differences for each metric
2. Wilcoxon signed-rank test (p < 0.05)
3. Bootstrap 95% confidence interval on mean difference
4. Apply Bonferroni correction for multiple comparisons

### Multi-Model Comparison

1. Friedman test for overall significance
2. Nemenyi post-hoc test for pairwise differences

## Results Reporting

### Required Reporting

For each model-dataset pair:
1. Test set metrics (all applicable)
2. 95% confidence intervals (bootstrap)
3. Training curves (loss, metrics vs. epoch)
4. Number of parameters
5. Training time
6. Embedding dimension

### Comparison Tables

1. Per-dataset model ranking
2. Per-task aggregated ranking
3. Model size vs. performance scatter plot

## Reproducibility Checklist

- [ ] Random seed set (default: 42)
- [ ] Deterministic algorithms enabled
- [ ] Configuration files saved with results
- [ ] Environment captured (Python, PyTorch, CUDA versions)
- [ ] Git commit hash recorded
- [ ] All random number generators seeded
- [ ] DataLoader workers seeded