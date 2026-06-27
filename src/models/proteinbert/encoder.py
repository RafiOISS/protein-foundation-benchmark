"""ProteinBERT encoder — sequence and label encoding utilities.

Responsibilities:
  - validate amino-acid alphabet
  - encode sequences to integer IDs
  - encode SS3 labels to integer IDs
  - apply configurable padding
  - apply configurable truncation
  - deterministic batching

No TensorFlow logic. Pure numpy/Python.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...utils.logging import get_logger
from .constants import (
    AA_ALPHABET,
    AA_TO_ID,
    AA_TO_ID_EXTENDED,
    LABEL_TO_ID,
    ID_TO_LABEL,
    PAD_ID,
    UNKNOWN_TOKEN,
    UNKNOWN_ID,
)


logger = get_logger(__name__)


# ------------------------------------------------------------------
# Amino-acid validation
# ------------------------------------------------------------------

VALID_AA_SET = set(AA_ALPHABET)
VALID_AA_SET_EXTENDED = set(AA_TO_ID_EXTENDED.keys())


def validate_sequence(
    sequence: str,
    extended_alphabet: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Validate a protein sequence against the amino-acid alphabet.

    Args:
        sequence: Protein sequence string.
        extended_alphabet: If True, allow B, Z, X, U, O (default: False).

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not sequence:
        return False, "Empty sequence"

    valid_set = VALID_AA_SET_EXTENDED if extended_alphabet else VALID_AA_SET

    for i, char in enumerate(sequence.upper()):
        if char not in valid_set:
            return False, f"Invalid residue '{char}' at position {i}"

    return True, None


def validate_sequences(
    sequences: List[str],
    extended_alphabet: bool = False,
    raise_on_invalid: bool = True,
) -> List[Tuple[int, str]]:
    """Validate a list of sequences.

    Args:
        sequences: List of protein sequences.
        extended_alphabet: Allow extended alphabet.
        raise_on_invalid: If True, raise ValueError on first invalid sequence.

    Returns:
        List of (index, error_message) for invalid sequences.

    Raises:
        ValueError: If raise_on_invalid and any sequence is invalid.
    """
    errors = []
    for i, seq in enumerate(sequences):
        valid, msg = validate_sequence(seq, extended_alphabet)
        if not valid:
            errors.append((i, msg))
            if raise_on_invalid:
                raise ValueError(f"Sequence {i}: {msg}")

    return errors


# ------------------------------------------------------------------
# Sequence encoding
# ------------------------------------------------------------------


def encode_sequence(
    sequence: str,
    extended_alphabet: bool = False,
    unknown_token: str = UNKNOWN_TOKEN,
) -> List[int]:
    """Encode a protein sequence into integer IDs.

    Args:
        sequence: Protein sequence string.
        extended_alphabet: If True, use extended alphabet (25 chars).
        unknown_token: Token to use for unknown residues.

    Returns:
        List of integer IDs.
    """
    aa_to_id = AA_TO_ID_EXTENDED if extended_alphabet else AA_TO_ID
    seq_upper = sequence.upper()

    ids = []
    for char in seq_upper:
        if char in aa_to_id:
            ids.append(aa_to_id[char])
        elif extended_alphabet:
            ids.append(aa_to_id.get(unknown_token, UNKNOWN_ID))
        else:
            # Map to unknown if not in standard set but extended is available
            ids.append(AA_TO_ID_EXTENDED.get(char, AA_TO_ID_EXTENDED.get(unknown_token, UNKNOWN_ID)))

    return ids


def decode_sequence(
    ids: Union[List[int], np.ndarray],
    extended_alphabet: bool = False,
) -> str:
    """Decode integer IDs back to a protein sequence.

    Args:
        ids: List or array of integer IDs.
        extended_alphabet: If True, use extended alphabet.

    Returns:
        Decoded sequence string.
    """
    from .constants import ID_TO_AA, ID_TO_AA_EXTENDED

    id_to_aa = ID_TO_AA_EXTENDED if extended_alphabet else ID_TO_AA

    chars = []
    for idx in ids:
        if isinstance(idx, np.integer):
            idx = int(idx)
        if idx == PAD_ID:
            continue
        chars.append(id_to_aa.get(idx, UNKNOWN_TOKEN))

    return "".join(chars)


# ------------------------------------------------------------------
# Padding
# ------------------------------------------------------------------


def pad_sequences(
    sequences: List[List[int]],
    max_length: int,
    padding: str = "right",
    pad_value: int = PAD_ID,
) -> np.ndarray:
    """Pad encoded sequences to a fixed length.

    Args:
        sequences: List of encoded sequences (lists of ints).
        max_length: Target length.
        padding: 'right' (default) or 'left'.
        pad_value: Value to use for padding tokens.

    Returns:
        numpy array of shape (len(sequences), max_length).
    """
    result = np.full((len(sequences), max_length), pad_value, dtype=np.int32)

    for i, seq_ids in enumerate(sequences):
        length = min(len(seq_ids), max_length)
        if padding == "right":
            result[i, :length] = seq_ids[:length]
        elif padding == "left":
            result[i, -length:] = seq_ids[-length:]
        else:
            raise ValueError(f"Unknown padding strategy: '{padding}'")

    return result


def create_attention_mask(
    token_ids: np.ndarray,
    pad_value: int = PAD_ID,
) -> np.ndarray:
    """Create attention mask from token IDs (1 for real tokens, 0 for padding).

    Args:
        token_ids: Token ID array of shape (batch, seq_len).
        pad_value: Value used for padding tokens.

    Returns:
        Attention mask array of shape (batch, seq_len).
    """
    return (token_ids != pad_value).astype(np.int32)


# ------------------------------------------------------------------
# Truncation
# ------------------------------------------------------------------


def truncate_sequences(
    sequences: List[List[int]],
    max_length: int,
    strategy: str = "right",
) -> List[List[int]]:
    """Truncate encoded sequences to max_length.

    Args:
        sequences: List of encoded sequences.
        max_length: Maximum length.
        strategy: 'right' (truncate from end) or 'left' (truncate from start).

    Returns:
        List of truncated sequences.
    """
    if strategy == "right":
        return [seq[:max_length] for seq in sequences]
    elif strategy == "left":
        return [seq[-max_length:] for seq in sequences]
    else:
        raise ValueError(f"Unknown truncation strategy: '{strategy}'")


# ------------------------------------------------------------------
# Label encoding
# ------------------------------------------------------------------


def encode_label(label: str) -> int:
    """Encode a single SS3 label string to integer ID.

    Args:
        label: One of 'H', 'E', 'C'.

    Returns:
        Integer ID (0, 1, or 2).

    Raises:
        ValueError: If label is not recognized.
    """
    label_upper = label.upper().strip()
    if label_upper not in LABEL_TO_ID:
        raise ValueError(
            f"Unknown SS3 label: '{label}'. Must be one of {list(LABEL_TO_ID.keys())}"
        )
    return LABEL_TO_ID[label_upper]


def encode_labels(labels: List[str]) -> np.ndarray:
    """Encode a list of SS3 labels to integer IDs.

    Args:
        labels: List of label strings ('H', 'E', 'C').

    Returns:
        numpy array of shape (len(labels),).

    Raises:
        ValueError: If any label is unrecognized.
    """
    return np.array([encode_label(lbl) for lbl in labels], dtype=np.int32)


def decode_label(label_id: int) -> str:
    """Decode an integer SS3 label ID back to string.

    Args:
        label_id: Integer ID (0, 1, or 2).

    Returns:
        Label string ('H', 'E', or 'C').
    """
    if label_id not in ID_TO_LABEL:
        raise ValueError(f"Unknown SS3 label ID: {label_id}")
    return ID_TO_LABEL[label_id]


def decode_labels(label_ids: Union[List[int], np.ndarray]) -> List[str]:
    """Decode integer label IDs back to strings.

    Args:
        label_ids: List or array of integer IDs.

    Returns:
        List of label strings.
    """
    return [decode_label(int(idx)) for idx in label_ids]


# ------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------


def create_batches(
    sequences: List[str],
    labels: Optional[List[Any]] = None,
    batch_size: int = 8,
    max_length: int = 512,
    padding: str = "right",
    truncation: bool = True,
    extended_alphabet: bool = False,
    shuffle: bool = False,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict[str, Any]]:
    """Create batched, encoded, padded input from raw sequences.

    Args:
        sequences: Raw protein sequence strings.
        labels: Optional raw label values (strings for SS3).
        batch_size: Number of sequences per batch.
        max_length: Maximum sequence length for padding/truncation.
        padding: Padding strategy ('right', 'left').
        truncation: Whether to truncate sequences exceeding max_length.
        extended_alphabet: Use extended amino-acid alphabet.
        shuffle: Whether to shuffle the data.
        rng: Optional random state for shuffle.

    Returns:
        List of batch dicts with keys:
          - 'input_ids': (batch, max_length) int32 array
          - 'attention_mask': (batch, max_length) int32 array
          - 'labels': (batch,) int32 array (if labels provided)
          - 'lengths': (batch,) int32 array of original lengths
    """
    if rng is None:
        rng = np.random.RandomState(42)

    indices = list(range(len(sequences)))
    if shuffle:
        rng.shuffle(indices)

    batches = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]

        batch_seqs = [sequences[i] for i in batch_indices]
        batch_encoded = [encode_sequence(s, extended_alphabet) for s in batch_seqs]

        original_lengths = np.array([len(s) for s in batch_seqs], dtype=np.int32)

        if truncation:
            batch_encoded = truncate_sequences(batch_encoded, max_length)

        input_ids = pad_sequences(batch_encoded, max_length, padding)
        attention_mask = create_attention_mask(input_ids)

        batch_dict: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": original_lengths,
        }

        if labels is not None:
            batch_labels = [labels[i] for i in batch_indices]
            # Detect if labels are strings (SS3) or already ints
            if batch_labels and isinstance(batch_labels[0], str):
                batch_dict["labels"] = encode_labels(batch_labels)
            else:
                batch_dict["labels"] = np.array(batch_labels, dtype=np.int32)

        batches.append(batch_dict)

    return batches
