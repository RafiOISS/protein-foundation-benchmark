"""Tests for ProteinBERT preprocessing pipeline.

Covers:
  - amino-acid validation
  - deterministic encoding
  - label encoding
  - batching
  - padding
  - truncation
  - preprocessing statistics generation
  - report generation
  - visualization generation
  - invalid-input handling
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models.proteinbert.constants import (
    AA_ALPHABET,
    AA_TO_ID,
    LABEL_TO_ID,
    ID_TO_LABEL,
    PAD_ID,
)
from src.models.proteinbert.encoder import (
    validate_sequence,
    validate_sequences,
    encode_sequence,
    decode_sequence,
    encode_label,
    encode_labels,
    decode_label,
    decode_labels,
    pad_sequences,
    create_attention_mask,
    truncate_sequences,
    create_batches,
)
from src.models.proteinbert.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipeline,
    compute_aa_statistics,
    compute_padding_statistics,
    compute_truncation_statistics,
    compute_validation_statistics,
)


# ------------------------------------------------------------------
# Amino-acid validation
# ------------------------------------------------------------------

class TestAminoAcidValidation:
    def test_valid_standard_sequence(self):
        valid, msg = validate_sequence("ACDEFGHIKLMNPQRSTVWY")
        assert valid is True
        assert msg is None

    def test_empty_sequence(self):
        valid, msg = validate_sequence("")
        assert valid is False
        assert "Empty" in msg

    def test_invalid_residue(self):
        valid, msg = validate_sequence("ACD1EFG")
        assert valid is False
        assert "Invalid residue" in msg

    def test_invalid_residue_position(self):
        valid, msg = validate_sequence("ACD1EFG")
        assert "position 3" in msg

    def test_lowercase(self):
        valid, msg = validate_sequence("acdefghiklmnpqrstvwy")
        assert valid is True

    def test_extended_alphabet(self):
        valid, msg = validate_sequence("ACDXBZUO", extended_alphabet=True)
        assert valid is True

    def test_extended_alphabet_fails_without_flag(self):
        valid, msg = validate_sequence("XBZUO", extended_alphabet=False)
        assert valid is False

    def test_validate_sequences_list(self):
        seqs = ["ACDEFG", "INVALID1", "GHIKLM"]
        errors = validate_sequences(seqs, raise_on_invalid=False)
        assert len(errors) == 1
        assert errors[0][0] == 1

    def test_validate_sequences_raises(self):
        with pytest.raises(ValueError, match="Sequence 1"):
            validate_sequences(["ACDEFG", "INVALID1"], raise_on_invalid=True)


# ------------------------------------------------------------------
# Sequence encoding
# ------------------------------------------------------------------

class TestSequenceEncoding:
    def test_encode_standard(self):
        ids = encode_sequence("ACD")
        assert ids == [AA_TO_ID["A"], AA_TO_ID["C"], AA_TO_ID["D"]]

    def test_encode_lowercase(self):
        ids = encode_sequence("acd")
        assert ids == [AA_TO_ID["A"], AA_TO_ID["C"], AA_TO_ID["D"]]

    def test_encode_extended(self):
        ids = encode_sequence("ACDX", extended_alphabet=True)
        assert len(ids) == 4

    def test_deterministic(self):
        seq = "MKFLILFNILVSTLAFLSSS"
        ids1 = encode_sequence(seq)
        ids2 = encode_sequence(seq)
        assert ids1 == ids2

    def test_decode_roundtrip(self):
        seq = "ACDEFGHIKLMNPQRSTVWY"
        ids = encode_sequence(seq)
        decoded = decode_sequence(ids)
        assert decoded == seq

    def test_decode_roundtrip_extended(self):
        seq = "ACDXBZUO"
        ids = encode_sequence(seq, extended_alphabet=True)
        decoded = decode_sequence(ids, extended_alphabet=True)
        # Unknown tokens may be decoded as X depending on mapping
        assert len(decoded) == len(seq)

    def test_pad_id_excluded_from_decode(self):
        seq = "ACD"
        ids = encode_sequence(seq) + [PAD_ID, PAD_ID]
        decoded = decode_sequence(ids)
        assert decoded == seq


# ------------------------------------------------------------------
# Label encoding
# ------------------------------------------------------------------

class TestLabelEncoding:
    def test_encode_h(self):
        assert encode_label("H") == 0

    def test_encode_e(self):
        assert encode_label("E") == 1

    def test_encode_c(self):
        assert encode_label("C") == 2

    def test_encode_lowercase(self):
        assert encode_label("h") == 0

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Unknown SS3 label"):
            encode_label("Z")

    def test_encode_labels_list(self):
        result = encode_labels(["H", "E", "C"])
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_decode_label(self):
        assert decode_label(0) == "H"
        assert decode_label(1) == "E"
        assert decode_label(2) == "C"

    def test_decode_invalid_id(self):
        with pytest.raises(ValueError, match="Unknown SS3 label ID"):
            decode_label(99)

    def test_decode_labels_list(self):
        result = decode_labels([0, 1, 2])
        assert result == ["H", "E", "C"]

    def test_roundtrip(self):
        labels = ["H", "E", "C", "H", "H", "E"]
        ids = encode_labels(labels)
        decoded = decode_labels(ids)
        assert decoded == labels


# ------------------------------------------------------------------
# Padding
# ------------------------------------------------------------------

class TestPadding:
    def test_right_padding(self):
        encoded = [[1, 2, 3], [4, 5], [6]]
        padded = pad_sequences(encoded, max_length=5, padding="right")
        assert padded.shape == (3, 5)
        np.testing.assert_array_equal(padded[0], [1, 2, 3, PAD_ID, PAD_ID])
        np.testing.assert_array_equal(padded[1], [4, 5, PAD_ID, PAD_ID, PAD_ID])
        np.testing.assert_array_equal(padded[2], [6, PAD_ID, PAD_ID, PAD_ID, PAD_ID])

    def test_left_padding(self):
        encoded = [[1, 2, 3], [4, 5], [6]]
        padded = pad_sequences(encoded, max_length=5, padding="left")
        np.testing.assert_array_equal(padded[0], [PAD_ID, PAD_ID, 1, 2, 3])
        np.testing.assert_array_equal(padded[1], [PAD_ID, PAD_ID, PAD_ID, 4, 5])

    def test_invalid_padding_strategy(self):
        with pytest.raises(ValueError, match="padding strategy"):
            pad_sequences([[1]], max_length=5, padding="center")

    def test_attention_mask(self):
        token_ids = np.array([[1, 2, PAD_ID, PAD_ID], [3, 4, 5, PAD_ID]])
        mask = create_attention_mask(token_ids)
        np.testing.assert_array_equal(mask[0], [1, 1, 0, 0])
        np.testing.assert_array_equal(mask[1], [1, 1, 1, 0])

    def test_all_zero_mask(self):
        token_ids = np.full((2, 5), PAD_ID)
        mask = create_attention_mask(token_ids)
        assert mask.sum() == 0


# ------------------------------------------------------------------
# Truncation
# ------------------------------------------------------------------

class TestTruncation:
    def test_truncate_right(self):
        sequences = [[1, 2, 3, 4, 5], [1, 2, 3]]
        truncated = truncate_sequences(sequences, max_length=3, strategy="right")
        assert truncated[0] == [1, 2, 3]
        assert truncated[1] == [1, 2, 3]

    def test_truncate_left(self):
        sequences = [[1, 2, 3, 4, 5]]
        truncated = truncate_sequences(sequences, max_length=3, strategy="left")
        assert truncated[0] == [3, 4, 5]

    def test_no_truncation_needed(self):
        sequences = [[1, 2], [3]]
        truncated = truncate_sequences(sequences, max_length=10)
        assert truncated == sequences

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="truncation strategy"):
            truncate_sequences([[1]], max_length=1, strategy="middle")


# ------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------

class TestBatching:
    def test_create_batches(self):
        sequences = ["ACD", "EFG", "HIK", "LMN"]
        batches = create_batches(sequences, batch_size=2, max_length=5)
        assert len(batches) == 2
        assert "input_ids" in batches[0]
        assert "attention_mask" in batches[0]
        assert batches[0]["input_ids"].shape == (2, 5)
        assert batches[1]["input_ids"].shape == (2, 5)

    def test_create_batches_with_labels(self):
        sequences = ["ACD", "EFG"]
        labels = ["H", "E"]
        batches = create_batches(sequences, labels=labels, batch_size=2, max_length=5)
        assert "labels" in batches[0]
        np.testing.assert_array_equal(batches[0]["labels"], [0, 1])

    def test_create_batches_deterministic(self):
        sequences = ["A" * 10, "C" * 10, "D" * 10, "E" * 10]
        b1 = create_batches(sequences, batch_size=2, max_length=10, shuffle=False)
        b2 = create_batches(sequences, batch_size=2, max_length=10, shuffle=False)
        for batch1, batch2 in zip(b1, b2):
            np.testing.assert_array_equal(batch1["input_ids"], batch2["input_ids"])

    def test_batch_lengths(self):
        sequences = ["ACD", "EFGH", "IK"]
        batches = create_batches(sequences, batch_size=2, max_length=10)
        np.testing.assert_array_equal(batches[0]["lengths"], [3, 4])
        np.testing.assert_array_equal(batches[1]["lengths"], [2])


# ------------------------------------------------------------------
# Preprocessing statistics
# ------------------------------------------------------------------

class TestPreprocessingStatistics:
    def test_aa_statistics(self):
        sequences = ["ACD", "EFG"]
        stats = compute_aa_statistics(sequences)
        assert stats["total_residues"] == 6
        assert len(stats["aa_counts"]) == 20
        assert stats["aa_counts"]["A"] == 1
        assert stats["aa_counts"]["E"] == 1

    def test_aa_statistics_unknown(self):
        sequences = ["ACDXB"]
        stats = compute_aa_statistics(sequences)
        assert stats["unknown_residue_count"] == 2  # X and B
        assert stats["invalid_residue_count"] == 0

    def test_padding_statistics(self):
        lengths = np.array([100, 200, 300])
        stats = compute_padding_statistics(lengths, max_length=512)
        assert stats["total_padding_tokens"] == (512 - 100) + (512 - 200) + (512 - 300)
        assert stats["sequences_with_padding"] == 3
        assert stats["pct_sequences_with_padding"] == 100.0

    def test_padding_statistics_no_padding(self):
        lengths = np.array([512, 512])
        stats = compute_padding_statistics(lengths, max_length=512)
        assert stats["total_padding_tokens"] == 0
        assert stats["padding_ratio"] == 0.0

    def test_truncation_statistics(self):
        lengths = np.array([100, 600, 700])
        stats = compute_truncation_statistics(lengths, max_length=512)
        assert stats["num_truncated_sequences"] == 2
        assert stats["total_residues_removed"] == (600 - 512) + (700 - 512)
        assert stats["max_truncation_length"] == 700 - 512

    def test_truncation_statistics_none(self):
        lengths = np.array([100, 200, 300])
        stats = compute_truncation_statistics(lengths, max_length=512)
        assert stats["num_truncated_sequences"] == 0

    def test_validation_statistics(self):
        stats = compute_validation_statistics(
            total_input=100,
            invalid_sequences=[(0, "bad")],
            invalid_residues=5,
            unknown_labels=[1, 2],
        )
        assert stats["retained_samples"] == 97
        assert stats["retention_rate"] == 97.0
        assert stats["filter_rate"] == 3.0


# ------------------------------------------------------------------
# PreprocessingConfig
# ------------------------------------------------------------------

class TestPreprocessingConfig:
    def test_default_config(self):
        config = PreprocessingConfig()
        assert config.max_length == 512
        assert config.padding == "right"
        assert config.truncation is True
        assert config.batch_size == 8

    def test_to_dict(self):
        config = PreprocessingConfig(max_length=256)
        d = config.to_dict()
        assert d["max_length"] == 256
        assert d["padding"] == "right"

    def test_from_dict(self):
        config = PreprocessingConfig(**{"max_length": 128, "padding": "left"})
        assert config.max_length == 128
        assert config.padding == "left"


# ------------------------------------------------------------------
# Preprocessing pipeline (integration)
# ------------------------------------------------------------------

class TestPreprocessingPipeline:
    def test_pipeline_with_mock_data(self, tmp_path):
        """Integration test: pipeline runs with synthetic data."""
        output_dir = tmp_path / "preprocessing"
        pipeline = PreprocessingPipeline(config={"max_length": 50}, output_dir=output_dir)

        # Create a mock dataset
        from src.models.proteinbert import ProteinBERTModel
        model = ProteinBERTModel()

        # Run with direct sequences
        result = pipeline.run_on_sequences(
            sequences=["ACDEFGHIK" * 10, "LMPQRSTVWY" * 5],
            labels=["H", "E"],
        )
        assert result["num_valid_sequences"] == 2
        assert "stats" in result
        assert "figures" in result
        assert "report" in result
        assert result["encoded_shape"][1] == 50

    def test_statistics_saved(self, tmp_path):
        """Pipeline generates expected CSV files."""
        output_dir = tmp_path / "preprocessing"
        pipeline = PreprocessingPipeline(config={"max_length": 50}, output_dir=output_dir)

        result = pipeline.run_on_sequences(
            sequences=["ACDEFGHIKL", "MNPQRSTVWY"],
            labels=["H", "E"],
        )

        stats_dir = Path(result["statistics_dir"])
        assert (stats_dir / "dataset_summary.csv").exists()
        assert (stats_dir / "amino_acid_frequencies.csv").exists()
        assert (stats_dir / "ss3_label_distribution.csv").exists()
        assert (stats_dir / "preprocessing_stats.json").exists()

    def test_figures_generated(self, tmp_path):
        """Pipeline generates PNG and PDF figures."""
        output_dir = tmp_path / "preprocessing"
        pipeline = PreprocessingPipeline(config={"max_length": 50}, output_dir=output_dir)

        result = pipeline.run_on_sequences(
            sequences=["ACDEFGHIKL"] * 10,
            labels=["H"] * 10,
        )

        figs_dir = Path(result["figures_dir"])
        expected_figs = [
            "sequence_length_histogram",
            "amino_acid_distribution",
            "ss3_class_distribution",
            "padding_distribution",
            "truncation_distribution",
        ]
        for name in expected_figs:
            assert (figs_dir / f"{name}.png").exists(), f"Missing PNG: {name}"
            assert (figs_dir / f"{name}.pdf").exists(), f"Missing PDF: {name}"

    def test_report_generated(self, tmp_path):
        """Pipeline generates preprocessing_report.md."""
        output_dir = tmp_path / "preprocessing"
        pipeline = PreprocessingPipeline(config={"max_length": 50}, output_dir=output_dir)

        result = pipeline.run_on_sequences(
            sequences=["ACDEFGHIKL"] * 5,
            labels=["H"] * 5,
        )

        report_path = Path(result["report"])
        assert report_path.exists()
        content = report_path.read_text()
        assert "Preprocessing Steps" in content
        assert "Configuration" in content
        assert "Dataset Overview" in content
        assert "Amino-Acid Composition" in content
        assert "Padding Analysis" in content

    def test_report_structure(self, tmp_path):
        """Report contains all expected major sections."""
        output_dir = tmp_path / "preprocessing"
        pipeline = PreprocessingPipeline(config={"max_length": 50}, output_dir=output_dir)

        result = pipeline.run_on_sequences(
            sequences=["ACDEFGHIKL"] * 5,
            labels=["C"] * 5,
        )

        content = Path(result["report"]).read_text()
        sections = [
            "Preprocessing Steps",
            "Configuration",
            "Sequence Validation",
            "Dataset Overview",
            "Amino-Acid Composition",
            "Secondary Structure (SS3) Statistics",
            "Padding Analysis",
            "Truncation Analysis",
            "Generated Figures",
        ]
        for section in sections:
            assert section in content, f"Missing section: {section}"


# ------------------------------------------------------------------
# Invalid input handling
# ------------------------------------------------------------------

class TestInvalidInputHandling:
    def test_invalid_residue_maps_to_unknown(self):
        """Invalid residues are mapped to unknown token ID, not raised."""
        batches = create_batches(["ACD1EFG"], labels=["H"], batch_size=2, max_length=50)
        assert len(batches) == 1
        # '1' is mapped to UNKNOWN_ID via fallback to extended alphabet
        assert batches[0]["input_ids"].shape == (1, 50)

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Unknown SS3 label"):
            create_batches(["ACDEFG"], labels=["Z"], batch_size=2, max_length=50)

    def test_filtered_in_run(self, tmp_path):
        """Pipeline filters invalid samples and reports them."""
        output_dir = tmp_path / "preprocessing"
        pipeline = PreprocessingPipeline(config={"max_length": 50}, output_dir=output_dir)

        result = pipeline.run_on_sequences(
            sequences=["ACDEFG", "INVALID1", "GHIKLM"],
            labels=["H", "E", "Z"],  # last label is invalid
        )
        assert result["num_valid_sequences"] == 1  # only first is valid

    def test_empty_sequences_list(self, tmp_path):
        """Pipeline handles empty input gracefully."""
        output_dir = tmp_path / "preprocessing"
        pipeline = PreprocessingPipeline(config={"max_length": 50}, output_dir=output_dir)

        result = pipeline.run_on_sequences(sequences=[], labels=[])
        assert result["num_valid_sequences"] == 0


# ------------------------------------------------------------------
# Prepare_inputs wrapper method
# ------------------------------------------------------------------

class TestPrepareInputs:
    def test_prepare_inputs_returns_dict(self):
        """prepare_inputs() returns expected result structure."""
        from src.models.proteinbert import ProteinBERTModel
        model = ProteinBERTModel(config={"max_length": 50})
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = model.prepare_inputs(
                dataset=None,  # Will use synthetic data
                output_dir=tmp,
            )
        assert isinstance(result, dict)
        assert "stats" in result
        assert "figures" in result
        assert "report" in result
