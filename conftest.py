"""Ensure the repository root is importable as `src`, `config`, `dashboard` when running pytest."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
