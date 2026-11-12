#### Preamble ####
# Purpose: Make an interactive 3D viewer page for a molecule, to check its stereocentres by eye
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: none (the page opens in any web browser)
# A helper, not part of the numbered pipeline. Writes other/viewer/<name>.html. Give the SMILES with stereo marks on
# the centres you want to check; marks RDKit would remove (e.g. on aryl nitrogens) are kept.
#
# Examples:
#   uv run scripts/view_molecule.py "C[C@H](O)c1ccccc1" --name "(S)-1-phenylethanol" --open
#   uv run scripts/view_molecule.py "Cc1ccc2c(c1)C[N@@]1C[N@]2Cc2cc(C)ccc21" --name "Tröger's base" --open

import argparse
import re
import subprocess
import unicodedata

from smiley_bench import paths
from smiley_bench.viewer import build_model, page_html

parser = argparse.ArgumentParser(description="Make an interactive 3D viewer page for a molecule.")
parser.add_argument("smiles", help="the molecule, with stereo marks on the centres to check")
parser.add_argument("--name", default="Molecule", help="a name for the page title and file name")
parser.add_argument("--open", action="store_true", help="open the page in the default browser")
args = parser.parse_args()

data = build_model(args.smiles, args.name)
plain_name = unicodedata.normalize("NFKD", args.name).encode("ascii", "ignore").decode()  # Tröger -> Troger
file_name = re.sub(r"[^A-Za-z0-9]+", "-", plain_name).strip("-").lower() or "molecule"
output = paths.ROOT / "other" / "viewer" / f"{file_name}.html"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(page_html(data))

print(f"Saved {output}")
for centre in data["centres"]:
    rdkit = f"RDKit: {centre['rdkit']}" if centre["rdkit"] else "RDKit: no label"
    tie = " (tie on atomic numbers: check by hand)" if centre["tie"] else ""
    print(f"  {centre['name']}: {centre['label']} from the model; {rdkit}{tie}")
if args.open:
    subprocess.run(["open", str(output)], check=False)  # macOS; on other systems open the file by hand
