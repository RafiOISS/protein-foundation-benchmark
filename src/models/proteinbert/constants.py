"""ProteinBERT constants — amino-acid alphabet, label mappings, and config defaults.

No TensorFlow imports. Pure Python.
"""

from typing import Dict, List

# ------------------------------------------------------------------
# Standard amino-acid alphabet (20 canonical + 2 non-standard + unknown)
# ------------------------------------------------------------------

AA_ALPHABET: List[str] = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]

AA_EXTENDED: List[str] = AA_ALPHABET + ["B", "Z", "X", "U", "O"]

AA_TO_ID: Dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}
AA_TO_ID_EXTENDED: Dict[str, int] = {aa: i for i, aa in enumerate(AA_EXTENDED)}

ID_TO_AA: Dict[int, str] = {i: aa for aa, i in AA_TO_ID.items()}
ID_TO_AA_EXTENDED: Dict[int, str] = {i: aa for aa, i in AA_TO_ID_EXTENDED.items()}

NUM_STANDARD_AAS: int = len(AA_ALPHABET)

# Special tokens
PAD_TOKEN: str = "<PAD>"
PAD_ID: int = len(AA_EXTENDED)  # 25
UNKNOWN_TOKEN: str = "X"
UNKNOWN_ID: int = AA_TO_ID_EXTENDED.get("X", 22)

# ------------------------------------------------------------------
# SS3 label mappings (3-class secondary structure)
# ------------------------------------------------------------------

LABEL_TO_ID: Dict[str, int] = {
    "H": 0,  # Helix
    "E": 1,  # Strand
    "C": 2,  # Coil
}

ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}

LABEL_NAMES: Dict[int, str] = {
    0: "Helix (H)",
    1: "Strand (E)",
    2: "Coil (C)",
}

NUM_SS3_CLASSES: int = 3

# ------------------------------------------------------------------
# Default preprocessing configuration
# ------------------------------------------------------------------

DEFAULT_PREPROCESSING_CONFIG = {
    "padding": "right",
    "truncation": True,
    "mask_padding": True,
    "max_length": 512,
    "unknown_token": "X",
    "alphabet": "standard",  # "standard" (20 AA) or "extended" (25 AA)
}
