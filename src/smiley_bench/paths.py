#### Preamble ####
# Purpose: The project's folders, worked out from where this file is, so scripts run from any folder
# Author: Sean Murphy
# Date: 11 November 2026
# Pre-requisites: none

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/smiley_bench/paths.py -> project root

SIMULATED_DATA = ROOT / "data" / "00-simulated_data"
RAW_DATA = ROOT / "data" / "01-raw_data"
ANALYSIS_DATA = ROOT / "data" / "02-analysis_data"

RESPONSES = RAW_DATA / "responses"  # one JSON file per model response
