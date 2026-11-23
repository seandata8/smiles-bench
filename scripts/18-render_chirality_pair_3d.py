#### Preamble ####
# Purpose: Rotating 3D models of two more molecules from the chirality test, side by side, for the website
#          (Discussion, "Trouble with Chirality"): trovan-tropine-deriv (achiral, meso) and spiro-heptane-F (chiral)
# Author: Sean Murphy
# Date: 23 November 2026
# Pre-requisites: an internet connection to view the page (it loads 3Dmol.js from a CDN)
# Each model is built from the benchmark's first SMILES form by model_3d() (force-field minimised, every stereo mark
# checked against the SMILES). Drawn as in scripts/14-render_3d.py: sticks, the page's background, slow spin, not
# turned or zoomed by hand; hydrogens left out. Saves website/hardest/_chirality-pair-3d.qmd.
#
#   uv run scripts/18-render_chirality_pair_3d.py

import json

import polars as pl
from rdkit import Chem

from smiley_bench import paths
from smiley_bench.stereo3d import model_3d
from smiley_bench.tables import read_with_canary

MOLECULES = ["trovan-tropine-deriv", "spiro-heptane-F"]  # left, right
PAGE = "#fbf6e6"  # the page's background (website/styles.css)
SPIN_SPEED = 0.3  # 3Dmol's units; slow, as in scripts/14-render_3d.py
JS = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"

tests = read_with_canary(paths.RAW_DATA / "molecule_tests.csv").filter(pl.col("test") == "chirality")
blocks = {}
for name in MOLECULES:
    smiles = tests.filter(pl.col("name") == name)["smiles_1"][0]
    blocks[name] = Chem.MolToMolBlock(Chem.RemoveHs(model_3d(smiles)))
    print(f"{name}: built")

figures = "\n".join(f'<figure><div id="{name}-3d" class="viewer-3d"></div><figcaption>{name}</figcaption></figure>'
                    for name in MOLECULES)
viewers = "\n".join(f"  show(\"{name}-3d\", {json.dumps(blocks[name])});" for name in MOLECULES)
page = paths.ROOT / "website" / "hardest" / "_chirality-pair-3d.qmd"
page.write_text(f"""```{{=html}}
<div class="molecule-pair">
{figures}
</div>
<script src="{JS}"></script>
<script>
(function () {{
  function show(id, block) {{
    const viewer = $3Dmol.createViewer(document.getElementById(id), {{backgroundColor: "{PAGE}", nomouse: true}});
    viewer.addModel(block, "sdf");
    viewer.setStyle({{}}, {{stick: {{radius: 0.18}}}});
    viewer.zoomTo(); viewer.render(); viewer.spin("y", {SPIN_SPEED});
  }}
{viewers}
}})();
</script>
```
""")
print(f"Saved {page.relative_to(paths.ROOT)}")
