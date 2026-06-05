"""Central project path configuration.

This module defines standard folders and file locations used by the pipeline
packages. It avoids ad hoc path manipulation across the repository.

Directory structure:
    data/
      ├── support/    # support_research outputs
      ├── exposure/   # exposure_research outputs
      ├── diversity/  # similarity_research outputs (future)
      ├── clustering/ # final_model outputs (future)
      └── viz/        # final visualization outputs
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Centralized data folder
DATA_ROOT = REPO_ROOT / "data"

# Support outputs
SUPPORT_DATA_DIR = DATA_ROOT / "support"

# Exposure outputs
EXPOSURE_DATA_DIR = DATA_ROOT / "exposure"

# Visualization outputs
VIZ_DATA_DIR = DATA_ROOT / "viz"

# BigQuery raw outputs
BIGQUERY_DATA_DIR = DATA_ROOT / "bigquery"

# Similarity Outputs
SIMILARITY_DATA_DIR = DATA_ROOT / "similarity"

# Complexity Outputs
COMPLEXITY_DATA_DIR = DATA_ROOT / "complexity"

# Google Fonts Directory Location
# Can be overridden with the environment variable `GOOGLE_FONTS_DIR`.
# Example:
#   export GOOGLE_FONTS_DIR="$HOME/google/fonts"
_env_google = os.environ.get("GOOGLE_FONTS_DIR")
if _env_google:
  GOOGLE_FONTS_DIR = Path(_env_google).expanduser()
else:
  GOOGLE_FONTS_DIR = REPO_ROOT / "similarity_research" / "diversity_research" / "fonts"


# Legacy module roots (for reference; not actively used in refactored code)
SUPPORT_ROOT = REPO_ROOT / "support_research"
EXPOSURE_ROOT = REPO_ROOT / "exposure_research"
DATA_VIZ_ROOT = REPO_ROOT / "data_viz"