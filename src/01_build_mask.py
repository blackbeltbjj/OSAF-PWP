# -*- coding: utf-8 -*-
"""
===============================================================================
PROJECT
    Pacific Warm Pool (PWP) Centroid Analysis

SCRIPT
    01_build_mask.py

PURPOSE
    Read the original IndonesiaNew0.msk file and convert it into a NumPy mask.

INPUT
    data/masks/IndonesiaNew0.msk

OUTPUT
    data/processed/pacific_mask.npy
    outputs/figures/pacific_mask.png

AUTHOR
    Fabio Vieira Machado
===============================================================================
"""

import sys
from pathlib import Path

# Adiciona a raiz do projeto (pwp_mascara) ao sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import numpy as np
import matplotlib.pyplot as plt

# Importa as configurações globais do arquivo mestre
from config.config import *

# =============================================================================
# PATHS
# =============================================================================

MASK_FILE = MASK_DIR / "IndonesiaNew0.msk"

OUTPUT_MASK = PROCESSED_DIR / "pacific_mask.npy"
OUTPUT_FIGURE = FIGURE_DIR / "pacific_mask.png"

# Garantir diretórios de saída
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# READ ASCII MASK
# =============================================================================

rows = []

with open(MASK_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if len(line) == 0:
            continue
        rows.append([int(c) for c in line])

mask = np.array(rows, dtype=np.uint8)

# =============================================================================
# REPORT
# =============================================================================

print("=" * 70)
print("PACIFIC MASK")
print("=" * 70)

print(f"Rows              : {mask.shape[0]}")
print(f"Columns           : {mask.shape[1]}")
print(f"Shape             : {mask.shape}")

print(f"Ocean cells (1)   : {np.sum(mask==1):,}")
print(f"Land cells (0)    : {np.sum(mask==0):,}")

# =============================================================================
# SAVE NUMPY VERSION
# =============================================================================

np.save(OUTPUT_MASK, mask)

print()
print("Mask saved to:")
print(OUTPUT_MASK)

# =============================================================================
# QUICK FIGURE
# =============================================================================

plt.figure(figsize=(5, 10))

plt.imshow(
    mask,
    origin="upper",
    interpolation="nearest",
    aspect="auto",
    cmap="Blues"
)

plt.title("IndonesiaNew0.msk")

plt.xlabel("Mask column")
plt.ylabel("Mask row")

plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300
)

plt.close()

print(f"Figure saved to:")
print(OUTPUT_FIGURE)

print("=" * 70)
print("Finished.")
print("=" * 70)