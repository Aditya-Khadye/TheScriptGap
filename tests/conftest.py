import sys
from pathlib import Path

# Make the repo root importable (paths.py, analysis/, exposure_research/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
