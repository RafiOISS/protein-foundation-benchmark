"""Tests for ProteinBERT data adapter.

Covers:
  - adapter creation and configuration
  - create_batches: identity, determinism, labels, no labels
  - from_batches: loading precomputed batches
  - from_preprocessed: integration with preprocessing output
  - from_raw: convenience method
  - TensorValidator: unit tests for all validation functions
  - edge cases: empty dataset, single sample, varying batch sizes
  - metadata generation, content, and persistence
  - iteration protocol (len, getitem, iter, next)
  - wrapper integration (prepare_dataset)
  - error handling on invalid/malformed inputs
  - reset
  - no TF import at module level
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.proteinbert.adapter import (
    ProteinBERTDataAdapter,
    AdapterMetadata,
    TensorValidator,
    AdapterError,
    TensorValidationError,
)
from src.models.proteinbert import ProteinBERTModel
from src.models.proteinbert.encoder import create_batches as encoder_create_batches
from src.models.proteinbert.adapter import (
    _generate_adapter_report,
    _generate_tensor_shapes,
    _generate_batch_summary,
    _generate_adapter_statistics,
    _build_experiment_path,
)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

PAD_ID = 25
SAMPLE_SEQUENCES = ["ACDEFGHIKLMNPQRSTVWY", "ACD" * 50, "EFG" * 100]
SAMPLE_LABELS = ["H", "E", "C"]


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def adapter():
    return ProteinBERTDataAdapter()


@pytest.fixture
def configured_adapter():
    return ProteinBERTDataAdapter(config={"batch_size": 2, "max_length": 128})


@pytest.fixture
def sample_batches():
    return encoder_create_batches(
        sequences=SAMPLE_SEQUENCES,
        labels=SAMPLE_LABELS,
        batch_size=2,
        max_length=512,
    )


@pytest.fixture
def sample_batches_no_labels():
    return encoder_create_batches(
        sequences=SAMPLE_SEQUENCES,
        batch_size=2,
        max_length=512,
    )


# ------------------------------------------------------------------
# Adapter creation and configuration
# ------------------------------------------------------------------


class TestAdapterCreation:
    def test_default_config(self):
        a = ProteinBERTDataAdapter()
        assert a.config["batch_size"] == 8
        assert a.config["max_length"] == 512
        assert a.config["padding"] == "right"
        assert a.config["truncation"] is True
        assert a.config["shuffle"] is False
        assert a.config["seed"] == 42

    def test_custom_config(self):
        a = ProteinBERTDataAdapter(config={"batch_size": 16, "max_length": 256})
        assert a.config["batch_size"] == 16
        assert a.config["max_length"] == 256
        assert a.config["padding"] == "right"

    def test_config_override_partial(self):
        a = ProteinBERTDataAdapter(config={"batch_size": 4})
        assert a.config["batch_size"] == 4
        assert a.config["max_length"] == 512  # unchanged default

    def test_repr(self):
        a = ProteinBERTDataAdapter()
        assert "ProteinBERTDataAdapter" in repr(a)
        assert "0" in repr(a)

    def test_info_empty(self):
        a = ProteinBERTDataAdapter()
        info = a.info()
        assert info["num_batches"] == 0
        assert info["num_samples"] == 0
        assert info["validated"] is False

    def test_info_loaded(self, adapter, sample_batches):
        adapter.from_batches(sample_batches)
        info = adapter.info()
        assert info["num_batches"] == 2
        assert info["num_samples"] == 3
        assert info["validated"] is True


# ------------------------------------------------------------------
# create_batches
# ------------------------------------------------------------------


class TestCreateBatches:
    def test_creates_correct_number_of_batches(self, configured_adapter):
        batches = configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert len(batches) == 2  # 3 sequences, batch_size=2

    def test_batch_dict_keys(self, configured_adapter):
        batches = configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        for batch in batches:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            assert "lengths" in batch

    def test_batch_dict_keys_no_labels(self, configured_adapter):
        batches = configured_adapter.create_batches(SAMPLE_SEQUENCES)
        for batch in batches:
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" not in batch
            assert "lengths" in batch

    def test_input_ids_shape(self, configured_adapter):
        batches = configured_adapter.create_batches(SAMPLE_SEQUENCES)
        for batch in batches:
            assert batch["input_ids"].ndim == 2
            assert batch["input_ids"].shape[1] == 128  # max_length

    def test_deterministic(self, configured_adapter):
        a1 = configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        a2 = configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert len(a1) == len(a2)
        for b1, b2 in zip(a1, a2):
            np.testing.assert_array_equal(b1["input_ids"], b2["input_ids"])
            np.testing.assert_array_equal(b1["labels"], b2["labels"])

    def test_deterministic_different_seeds(self):
        a1 = ProteinBERTDataAdapter(config={"batch_size": 2, "seed": 1})
        a2 = ProteinBERTDataAdapter(config={"batch_size": 2, "seed": 2})
        batches1 = a1.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        batches2 = a2.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        # With shuffle=False (default), seeds don't matter
        np.testing.assert_array_equal(batches1[0]["labels"], batches2[0]["labels"])

    def test_labels_as_integers(self, configured_adapter):
        int_labels = [0, 1, 2]
        batches = configured_adapter.create_batches(SAMPLE_SEQUENCES, int_labels)
        np.testing.assert_array_equal(batches[0]["labels"][:2], [0, 1])
        np.testing.assert_array_equal(batches[1]["labels"][0], [2])

    def test_auto_validates(self, configured_adapter):
        batches = configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert configured_adapter._validated is True
        assert configured_adapter._num_batches == len(batches)
        assert configured_adapter._num_samples == 3

    def test_pad_id_correct(self, configured_adapter):
        batches = configured_adapter.create_batches(["ACD"], ["H"])
        token_ids = batches[0]["input_ids"]
        # A=0, C=1, D=2, rest should be PAD_ID (25)
        assert token_ids[0, 0] == 0
        assert token_ids[0, 1] == 1
        assert token_ids[0, 2] == 2
        assert np.all(token_ids[0, 3:] == PAD_ID)


# ------------------------------------------------------------------
# from_batches
# ------------------------------------------------------------------


class TestFromBatches:
    def test_accepts_encoder_batches(self, adapter, sample_batches):
        adapter.from_batches(sample_batches)
        assert adapter._num_batches == 2
        assert adapter._num_samples == 3

    def test_validates_on_load(self, adapter, sample_batches):
        adapter.from_batches(sample_batches)
        assert adapter._validated is True

    def test_rejects_empty_list(self, adapter):
        """Empty batch list is allowed — creates an empty adapter."""
        adapter.from_batches([])
        assert adapter._num_batches == 0
        assert adapter._num_samples == 0
        assert adapter._validated is True

    def test_rejects_missing_key(self, adapter, sample_batches):
        del sample_batches[0]["input_ids"]
        with pytest.raises(TensorValidationError, match="missing required key"):
            adapter.from_batches(sample_batches)

    def test_rejects_mismatched_batch_dim(self, adapter):
        # Within a batch, input_ids has batch_size=2 but labels has batch_size=3
        bad_batch = {
            "input_ids": np.zeros((2, 512), dtype=np.int32),
            "attention_mask": np.ones((2, 512), dtype=np.int32),
            "labels": np.array([0, 1, 2], dtype=np.int32),  # mismatched
        }
        with pytest.raises(TensorValidationError, match="!= batch_size"):
            adapter.from_batches([bad_batch])

    def test_rejects_mismatched_seq_dim(self, adapter):
        batches = [
            {
                "input_ids": np.zeros((2, 128), dtype=np.int32),
                "attention_mask": np.ones((2, 512), dtype=np.int32),
            }
        ]
        with pytest.raises(TensorValidationError, match="seq dim"):
            adapter.from_batches(batches)

    def test_returns_self_for_chaining(self, adapter, sample_batches):
        result = adapter.from_batches(sample_batches)
        assert result is adapter


# ------------------------------------------------------------------
# from_preprocessed
# ------------------------------------------------------------------


class TestFromPreprocessed:
    def test_accepts_preprocessing_dict(self, adapter):
        prepared = {
            "encoded_shape": [3, 512],
            "stats": {
                "validation": {
                    "retained_samples": 3,
                }
            },
        }
        result = adapter.from_preprocessed(prepared)
        assert result is adapter

    def test_rejects_missing_encoded_shape(self, adapter):
        with pytest.raises(AdapterError, match="encoded_shape"):
            adapter.from_preprocessed({})

    def test_sets_max_length(self, adapter):
        prepared = {
            "encoded_shape": [3, 256],
            "stats": {},
        }
        adapter.from_preprocessed(prepared)
        assert adapter.config["max_length"] == 256


# ------------------------------------------------------------------
# from_raw
# ------------------------------------------------------------------


class TestFromRaw:
    def test_convenience_method(self, configured_adapter):
        result = configured_adapter.from_raw(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert result is configured_adapter
        assert configured_adapter._num_batches == 2
        assert configured_adapter._num_samples == 3
        assert configured_adapter._validated is True


# ------------------------------------------------------------------
# TensorValidator unit tests
# ------------------------------------------------------------------


class TestTensorValidator:
    def test_validate_input_ids_valid(self):
        tensor = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        TensorValidator.validate_input_ids(tensor, 0)  # should not raise

    def test_validate_input_ids_bad_type(self):
        with pytest.raises(TensorValidationError, match="input_ids must be np.ndarray"):
            TensorValidator.validate_input_ids([1, 2, 3], 0)

    def test_validate_input_ids_bad_ndim(self):
        with pytest.raises(TensorValidationError, match="ndim=3"):
            TensorValidator.validate_input_ids(np.zeros((1, 2, 3), dtype=np.int32), 0)

    def test_validate_input_ids_bad_dtype(self):
        with pytest.raises(TensorValidationError, match="dtype"):
            TensorValidator.validate_input_ids(np.zeros((1, 2), dtype=np.float64), 0)

    def test_validate_input_ids_empty(self):
        with pytest.raises(TensorValidationError, match="empty"):
            TensorValidator.validate_input_ids(np.array([[]], dtype=np.int32), 0)

    def test_validate_attention_mask_valid(self):
        tensor = np.ones((2, 10), dtype=np.int32)
        TensorValidator.validate_attention_mask(tensor, 0, (2, 10))

    def test_validate_attention_mask_wrong_shape(self):
        with pytest.raises(TensorValidationError, match="shape"):
            TensorValidator.validate_attention_mask(
                np.ones((2, 20), dtype=np.int32), 0, (2, 10)
            )

    def test_validate_attention_mask_invalid_values(self):
        with pytest.raises(TensorValidationError, match="outside"):
            TensorValidator.validate_attention_mask(
                np.array([[0, 1, 2]], dtype=np.int32), 0, (1, 3)
            )

    def test_validate_labels_valid(self):
        tensor = np.array([0, 1, 2], dtype=np.int32)
        TensorValidator.validate_labels(tensor, 0, 3)

    def test_validate_labels_wrong_ndim(self):
        with pytest.raises(TensorValidationError, match="expected 1D"):
            TensorValidator.validate_labels(np.zeros((1, 2), dtype=np.int32), 0, 1)

    def test_validate_labels_wrong_length(self):
        with pytest.raises(TensorValidationError, match="length.*num_samples"):
            TensorValidator.validate_labels(np.array([0, 1], dtype=np.int32), 0, 3)

    def test_validate_lengths_valid(self):
        tensor = np.array([10, 20, 30], dtype=np.int32)
        TensorValidator.validate_lengths(tensor, 0, 3)

    def test_validate_lengths_wrong_dtype(self):
        with pytest.raises(TensorValidationError, match="dtype"):
            TensorValidator.validate_lengths(np.array([1.0, 2.0]), 0, 2)

    def test_validate_batch_size_consistency_valid(self):
        batches = [
            {"input_ids": np.zeros((2, 10), dtype=np.int32), "attention_mask": np.ones((2, 10), dtype=np.int32)},
            {"input_ids": np.zeros((1, 10), dtype=np.int32), "attention_mask": np.ones((1, 10), dtype=np.int32)},
        ]
        assert TensorValidator.validate_batch_size_consistency(batches) == 2

    def test_validate_batch_size_consistency_mismatch(self):
        batches = [
            {"input_ids": np.zeros((2, 10), dtype=np.int32), "attention_mask": np.ones((3, 10), dtype=np.int32)},
        ]
        with pytest.raises(TensorValidationError, match="!= batch_size"):
            TensorValidator.validate_batch_size_consistency(batches)

    def test_validate_seq_dim_consistency_valid(self):
        batches = [
            {"input_ids": np.zeros((2, 128), dtype=np.int32), "attention_mask": np.ones((2, 128), dtype=np.int32)},
            {"input_ids": np.zeros((2, 128), dtype=np.int32), "attention_mask": np.ones((2, 128), dtype=np.int32)},
        ]
        assert TensorValidator.validate_seq_dim_consistency(batches) == 128

    def test_validate_seq_dim_consistency_mismatch(self):
        batches = [
            {"input_ids": np.zeros((2, 128), dtype=np.int32), "attention_mask": np.ones((2, 128), dtype=np.int32)},
            {"input_ids": np.zeros((2, 64), dtype=np.int32), "attention_mask": np.ones((2, 64), dtype=np.int32)},
        ]
        with pytest.raises(TensorValidationError, match="seq dim"):
            TensorValidator.validate_seq_dim_consistency(batches)

    def test_validate_mask_consistency_correct(self):
        ids = np.array([[1, 2, PAD_ID, PAD_ID], [3, 4, 5, PAD_ID]], dtype=np.int32)
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=np.int32)
        batches = [{"input_ids": ids, "attention_mask": mask}]
        TensorValidator.validate_mask_consistency(batches)  # should not raise

    def test_validate_mask_consistency_incorrect(self):
        ids = np.array([[1, PAD_ID, 3]], dtype=np.int32)
        mask = np.array([[1, 1, 1]], dtype=np.int32)  # should be [1, 0, 1]
        batches = [{"input_ids": ids, "attention_mask": mask}]
        with pytest.raises(TensorValidationError, match="inconsistent"):
            TensorValidator.validate_mask_consistency(batches)

    def test_validate_all(self, sample_batches):
        stats = TensorValidator.validate_all(sample_batches)
        assert stats["num_batches"] == 2
        assert stats["total_samples"] == 3
        assert stats["all_masks_consistent"] is True
        assert stats["mask_errors"] == 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_sequences(self, configured_adapter):
        batches = configured_adapter.create_batches([], [])
        assert len(batches) == 0

    def test_single_sample(self):
        a = ProteinBERTDataAdapter(config={"batch_size": 2, "max_length": 50})
        batches = a.create_batches(["ACDEFGHIKL"], ["H"])
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape[0] == 1

    def test_batch_size_equals_dataset(self):
        a = ProteinBERTDataAdapter(config={"batch_size": 3, "max_length": 128})
        batches = a.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape[0] == 3

    def test_batch_size_larger_than_dataset(self):
        a = ProteinBERTDataAdapter(config={"batch_size": 10, "max_length": 128})
        batches = a.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert len(batches) == 1
        assert batches[0]["input_ids"].shape[0] == 3

    def test_very_long_sequences(self):
        a = ProteinBERTDataAdapter(config={"batch_size": 2, "max_length": 50})
        long_seqs = ["A" * 200, "C" * 200]
        batches = a.create_batches(long_seqs, ["H", "E"])
        for batch in batches:
            assert batch["input_ids"].shape[1] == 50
            assert batch["attention_mask"].shape[1] == 50

    def test_varying_batch_size_last_batch(self):
        a = ProteinBERTDataAdapter(config={"batch_size": 4, "max_length": 128})
        seqs = ["A" * 10, "C" * 10, "D" * 10, "E" * 10, "F" * 10]
        batches = a.create_batches(seqs)
        assert len(batches) == 2  # 4 + 1
        assert batches[0]["input_ids"].shape[0] == 4
        assert batches[1]["input_ids"].shape[0] == 1


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------


class TestMetadata:
    def test_generated_after_batches(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta = configured_adapter.metadata
        assert meta["num_batches"] == 2
        assert meta["num_samples"] == 3
        assert meta["batch_size"] == 2
        assert meta["seq_len"] == 128

    def test_metadata_has_labels(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta = configured_adapter.metadata
        assert meta["has_labels"] is True
        assert meta["label_dim"] == 1

    def test_metadata_no_labels(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES)
        meta = configured_adapter.metadata
        assert meta["has_labels"] is False

    def test_metadata_padding_ratio(self, configured_adapter):
        configured_adapter.create_batches(["A"], ["H"])
        meta = configured_adapter.metadata
        assert meta["padding_ratio"] > 0.0  # 1 token out of 128
        assert meta["total_padding_tokens"] == 127

    def test_metadata_all_masks_consistent(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta = configured_adapter.metadata
        assert meta["all_masks_consistent"] is True

    def test_metadata_dtypes(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta = configured_adapter.metadata
        assert "int32" in meta["input_dtype"]
        assert "int32" in meta["label_dtype"]

    def test_metadata_timestamp(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta = configured_adapter.metadata
        assert meta["created_at"] != ""

    def test_save_metadata(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        path = configured_adapter.save_metadata(tmp_path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["num_batches"] == 2

    def test_generate_metadata_object(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta_obj = configured_adapter.generate_metadata()
        assert isinstance(meta_obj, AdapterMetadata)
        assert meta_obj.num_batches == 2

    def test_metadata_raises_if_no_batches(self, adapter):
        with pytest.raises(AdapterError, match="Cannot generate metadata"):
            adapter.generate_metadata()


# ------------------------------------------------------------------
# AdapterMetadata class
# ------------------------------------------------------------------


class TestAdapterMetadataClass:
    def test_from_batches(self, sample_batches):
        meta = AdapterMetadata.from_batches(sample_batches, {"max_length": 512})
        assert meta.num_batches == 2
        assert meta.num_samples == 3
        assert meta.batch_size == 2
        assert meta.seq_len == 512

    def test_to_dict(self, sample_batches):
        meta = AdapterMetadata.from_batches(sample_batches)
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["num_batches"] == 2

    def test_empty_batches(self):
        meta = AdapterMetadata.from_batches([])
        assert meta.num_batches == 0
        assert meta.num_samples == 0

    def test_defaults(self):
        meta = AdapterMetadata()
        assert meta.num_batches == 0
        assert meta.padding_ratio == 0.0


# ------------------------------------------------------------------
# Iteration
# ------------------------------------------------------------------


class TestIteration:
    def test_len(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert len(configured_adapter) == 2

    def test_getitem(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        batch = configured_adapter[0]
        assert "input_ids" in batch
        assert "labels" in batch

    def test_getitem_negative(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        batch = configured_adapter[-1]
        assert batch["input_ids"].shape[0] == 1  # last batch has 1 sample

    def test_iteration(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        count = 0
        for batch in configured_adapter:
            assert "input_ids" in batch
            count += 1
        assert count == 2

    def test_iteration_multiple_passes(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        count1 = sum(1 for _ in configured_adapter)
        count2 = sum(1 for _ in configured_adapter)
        assert count1 == 2
        assert count2 == 2

    def test_empty_iteration(self, adapter):
        adapter._batches = []
        adapter._validated = False
        assert len(adapter) == 0
        count = 0
        for _ in adapter:
            count += 1
        assert count == 0


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert configured_adapter._num_batches > 0
        assert configured_adapter._validated is True

        configured_adapter.reset()
        assert configured_adapter._num_batches == 0
        assert configured_adapter._num_samples == 0
        assert configured_adapter._validated is False

    def test_reset_allows_reuse(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        configured_adapter.reset()
        configured_adapter.create_batches(["A"], ["H"])
        assert len(configured_adapter) == 1


# ------------------------------------------------------------------
# Wrapper integration
# ------------------------------------------------------------------


class TestWrapperIntegration:
    def test_prepare_dataset_returns_adapter(self):
        model = ProteinBERTModel(config={"max_length": 50, "batch_size": 2})
        adapter = model.prepare_dataset(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert isinstance(adapter, ProteinBERTDataAdapter)
        assert len(adapter) == 2
        assert adapter.info()["num_samples"] == 3

    def test_prepare_dataset_with_output_dir(self, tmp_path):
        model = ProteinBERTModel(config={"max_length": 50, "batch_size": 2})
        adapter = model.prepare_dataset(
            SAMPLE_SEQUENCES, SAMPLE_LABELS, output_dir=tmp_path
        )
        assert (tmp_path / "adapter_metadata.json").exists()

    def test_prepare_dataset_no_labels(self):
        model = ProteinBERTModel(config={"max_length": 50, "batch_size": 2})
        adapter = model.prepare_dataset(SAMPLE_SEQUENCES)
        assert len(adapter) == 2
        assert adapter.info()["has_labels"] is False

    def test_prepare_dataset_validated(self):
        model = ProteinBERTModel(config={"max_length": 50, "batch_size": 2})
        adapter = model.prepare_dataset(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        assert adapter._validated is True

    def test_prepare_dataset_deterministic(self):
        model = ProteinBERTModel(config={"max_length": 50, "batch_size": 2})
        a1 = model.prepare_dataset(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        a2 = model.prepare_dataset(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        for b1, b2 in zip(a1, a2):
            np.testing.assert_array_equal(b1["input_ids"], b2["input_ids"])
            np.testing.assert_array_equal(b1["labels"], b2["labels"])


# ------------------------------------------------------------------
# TF Dataset conversion (lazy)
# ------------------------------------------------------------------


class TestTfDataset:
    def test_to_tf_dataset_requires_tf(self, configured_adapter):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        try:
            import tensorflow
            ds = configured_adapter.to_tf_dataset()
            assert ds is not None
        except ImportError:
            pytest.skip("TensorFlow not installed")


# ------------------------------------------------------------------
# Module-level imports
# ------------------------------------------------------------------


class TestModuleLevelImports:
    def test_no_tf_at_module_level(self):
        """Verify that importing the adapter does not import TensorFlow."""
        import sys
        tf_before = "tensorflow" in sys.modules
        from src.models.proteinbert.adapter import ProteinBERTDataAdapter
        tf_after = "tensorflow" in sys.modules
        assert tf_after == tf_before, (
            "Importing adapter imported TensorFlow at module level"
        )

    def test_adapter_exported_from_package(self):
        from src.models.proteinbert import ProteinBERTDataAdapter, AdapterMetadata, TensorValidator
        assert ProteinBERTDataAdapter is not None
        assert AdapterMetadata is not None
        assert TensorValidator is not None


# ------------------------------------------------------------------
# save_all
# ------------------------------------------------------------------


class TestSaveAll:
    def test_save_all_creates_all_files(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        artifacts = configured_adapter.save_all(output_dir=tmp_path)
        expected = [
            "adapter_report.md",
            "adapter_metadata.json",
            "adapter_statistics.json",
            "adapter_statistics.csv",
            "tensor_shapes.json",
            "batch_summary.csv",
        ]
        for name in expected:
            assert (tmp_path / name).exists(), f"Missing: {name}"
        assert len(artifacts) >= 6

    def test_save_all_report_content(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        configured_adapter.save_all(output_dir=tmp_path)
        report_content = (tmp_path / "adapter_report.md").read_text()
        assert "## Configuration" in report_content
        assert "## Dataset Overview" in report_content
        assert "## Tensor Shapes" in report_content
        assert "## Token Statistics" in report_content
        assert "## Mask Validation" in report_content
        assert "## Batch Summary" in report_content

    def test_save_all_tensor_shapes_content(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        configured_adapter.save_all(output_dir=tmp_path)
        import json
        with open(tmp_path / "tensor_shapes.json") as f:
            shapes = json.load(f)
        assert isinstance(shapes, list)
        assert len(shapes) == 2
        assert "batch_index" in shapes[0]
        assert "input_ids" in shapes[0]
        assert "attention_mask" in shapes[0]

    def test_save_all_batch_summary_content(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        configured_adapter.save_all(output_dir=tmp_path)
        import pandas as pd
        df = pd.read_csv(tmp_path / "batch_summary.csv")
        assert len(df) == 2
        assert "batch_index" in df.columns
        assert "samples" in df.columns
        assert "padding_pct" in df.columns

    def test_save_all_statistics_json(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        configured_adapter.save_all(output_dir=tmp_path)
        import json
        with open(tmp_path / "adapter_statistics.json") as f:
            stats = json.load(f)
        assert stats["num_batches"] == 2
        assert stats["num_samples"] == 3
        assert "padding_ratio" in stats
        assert "all_masks_consistent" in stats

    def test_save_all_statistics_csv(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        configured_adapter.save_all(output_dir=tmp_path)
        import pandas as pd
        df = pd.read_csv(tmp_path / "adapter_statistics.csv")
        assert len(df) == 1
        assert df["num_batches"].iloc[0] == 2

    def test_save_all_raises_if_no_batches(self, adapter, tmp_path):
        with pytest.raises(AdapterError, match="Cannot save"):
            adapter.save_all(output_dir=tmp_path)

    def test_save_all_with_experiment_id(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        # Use a subdirectory simulating experiment_id pattern
        out = tmp_path / "experiments" / "exp_001" / "adapter"
        configured_adapter.save_all(output_dir=out)
        assert (out / "adapter_report.md").exists()

    def test_save_all_no_labels(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES)
        configured_adapter.save_all(output_dir=tmp_path)
        import json
        with open(tmp_path / "tensor_shapes.json") as f:
            shapes = json.load(f)
        assert "labels" not in shapes[0]

    def test_save_all_single_batch(self, tmp_path):
        a = ProteinBERTDataAdapter(config={"batch_size": 10, "max_length": 50})
        a.create_batches(["ACD", "EFG", "HIK"])
        artifacts = a.save_all(output_dir=tmp_path)
        assert (tmp_path / "batch_summary.csv").exists()
        import pandas as pd
        df = pd.read_csv(tmp_path / "batch_summary.csv")
        assert df["samples"].iloc[0] == 3


# ------------------------------------------------------------------
# _build_experiment_path
# ------------------------------------------------------------------


class TestBuildExperimentPath:
    def test_with_output_dir(self):
        path = _build_experiment_path(output_dir="my/custom/path")
        assert path == Path("my/custom/path")

    def test_with_experiment_id(self):
        path = _build_experiment_path(experiment_id="exp_001")
        assert path == Path("outputs/experiments/exp_001/adapter")

    def test_default(self):
        path = _build_experiment_path()
        assert path == Path("outputs/adapter")

    def test_both_provided_output_dir_wins(self):
        path = _build_experiment_path(
            output_dir="custom/path",
            experiment_id="exp_001",
        )
        assert path == Path("custom/path")


# ------------------------------------------------------------------
# Module-level generation functions
# ------------------------------------------------------------------


class TestGenerationFunctions:
    def test_generate_adapter_report(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta = configured_adapter.generate_metadata()
        shapes = [{"batch_index": 0, "input_ids": {"shape": [2, 128], "ndim": 2, "dtype": "int32", "size": 256},
                    "attention_mask": {"shape": [2, 128], "ndim": 2, "dtype": "int32", "size": 256},
                    "batch_samples": 2, "total_tokens": 256, "padding_tokens": 200, "padding_pct": 78.12,
                    "mask_errors": 0},
                  {"batch_index": 1, "input_ids": {"shape": [1, 128], "ndim": 2, "dtype": "int32", "size": 128},
                    "attention_mask": {"shape": [1, 128], "ndim": 2, "dtype": "int32", "size": 128},
                    "batch_samples": 1, "total_tokens": 128, "padding_tokens": 100, "padding_pct": 78.12,
                    "mask_errors": 0}]
        report_path = _generate_adapter_report(
            meta, configured_adapter.config,
            configured_adapter._validation_stats, shapes,
            tmp_path / "adapter_report.md",
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "Adapter Report" in content

    def test_generate_tensor_shapes(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        path = _generate_tensor_shapes(configured_adapter._batches, tmp_path / "tensor_shapes.json")
        assert path.exists()
        import json
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert "padding_pct" in data[0]

    def test_generate_batch_summary(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        path = _generate_batch_summary(configured_adapter._batches, tmp_path / "batch_summary.csv")
        assert path.exists()
        import pandas as pd
        df = pd.read_csv(path)
        assert len(df) == 2

    def test_generate_adapter_statistics(self, configured_adapter, tmp_path):
        configured_adapter.create_batches(SAMPLE_SEQUENCES, SAMPLE_LABELS)
        meta = configured_adapter.generate_metadata()
        paths = _generate_adapter_statistics(
            meta, configured_adapter._validation_stats,
            tmp_path / "stats.json", tmp_path / "stats.csv",
        )
        assert paths["json"].exists()
        assert paths["csv"].exists()
        import json
        with open(paths["json"]) as f:
            stats = json.load(f)
        assert stats["num_batches"] == 2
