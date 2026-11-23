#### Preamble ####
# Purpose: One 3D model of the benchmark's maitotoxin derivative (too big to show two side by side), with Sean's
#          changes in clouds and labelled: the benzyl ether and the C=C added (blue), and the OH removed (red, where its
#          O was) (website Discussion, "Lost Count")
# Author: Sean Murphy
# Date: 23 November 2026
# Pre-requisites: PubChem's maitotoxin (CID 71460273, its 2D record) as data/01-raw_data/structures/website/
#                 maitotoxin.sdf; Google Chrome (macOS) and an internet connection (3Dmol.js loads from a CDN)
# The parent's 3D model is made from PubChem's 2D drawing (stretched out, as maitotoxin is usually drawn): lifted off
# the page by its wedges and minimised locally, so it stays stretched out (a model built from scratch folds up). The
# derivative is made from it, as in scripts/14-render_3d.py. The benzyl group and the C=C are found by pattern; the
# carbon that lost its OH is the one where adding an OH back (with the benzyl removed and the C=C made single) gives
# PubChem's maitotoxin, its sodium sulfates as the free acid (the fourth change, not shown). The red cloud is where
# the parent's OH oxygen was. The picture is drawn by 3Dmol.js in headless Chrome (turned so the molecule is longest
# across), cropped, and labelled with matplotlib (LABEL_PLACES). Saves other/figures/hardest/maitotoxin-deriv-3d.png.
#
#   uv run scripts/15-render_maitotoxin_3d.py

import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from PIL import Image
from rdkit.Geometry import Point3D

from smiley_bench import paths
from smiley_bench.benchmark import STRUCTURES
from smiley_bench.tables import read_with_canary

NAME = "maitotoxin-deriv"
RED, BLUE = "#f27373", "#73a8f2"  # as in the other molecules' figures
PAGE = "#fbf6e6"  # the page's background (website/styles.css): see-through clouds need a solid background
CLOUD = {"radius": 1.1, "opacity": 0.6}  # as in scripts/14-render_3d.py
JS = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"  # headless, to draw the picture (macOS)
VIEW = (1600, 1000)  # the drawing's size in the browser, in pixels
SCALE = 2  # pixels per browser pixel, for a sharp picture
DPI = 200
FONT_SIZE = 7.9
# Where each label goes: across and down the picture, as shares of its width and height, in empty space next to
# its cloud, so neither the label nor its line covers the molecule (set by eye from the picture)
LABEL_PLACES = {"Benzyl ether added": (0.70, 0.38), "C=C added": (0.45, 0.47), "OH removed": (0.60, 0.63)}
FIGURES = paths.ROOT / "other" / "figures" / "hardest"
WEDGE, HASH = 1, 6  # bond stereo in a mol file


def minimise(model, fixed=()):
    """MMFF, with a distance-dependent dielectric of 80 (as in water), holding the `fixed` atoms in place."""
    properties = AllChem.MMFFGetMoleculeProperties(model)
    properties.SetMMFFDielectricModel(1)  # distance-dependent
    properties.SetMMFFDielectricConstant(80.0)
    field = AllChem.MMFFGetMoleculeForceField(model, properties)
    for atom in fixed:
        field.AddFixedPoint(atom)
    field.Minimize(maxIts=20000)


def check_stereo(model, smiles):
    """An error unless the model's 3D shape has the stereocentres written in `smiles`."""
    check = Chem.Mol(model)
    Chem.AssignStereochemistryFrom3D(check)
    if Chem.MolToSmiles(Chem.RemoveHs(check)) != smiles:
        raise ValueError("the 3D model's stereocentres don't match the SMILES")


def flat(mol):
    """The molecule's SMILES without stereo: the atoms and bonds only."""
    return Chem.MolToSmiles(mol, isomericSmiles=False)


smiles = read_with_canary(paths.RAW_DATA / "molecule_tests.csv").filter(pl.col("name") == NAME)["smiles_1"][0]
mol = Chem.MolFromSmiles(smiles)
benzyl = mol.GetSubstructMatch(Chem.MolFromSmarts("[CH2;!R]c1ccccc1"))  # the benzyl's CH2 and ring
alkene = mol.GetSubstructMatches(Chem.MolFromSmarts("[CH1;r7]=[CH1;r7]"))
if len(alkene) != 1:
    raise ValueError(f"expected one C=C in a 7-membered ring, found {len(alkene)}")
alkene = alkene[0]

# The parent: PubChem's maitotoxin (its 2D record, laid out stretched out, as it is usually drawn)
parent_3d = Chem.MolFromMolFile(str(STRUCTURES / "website" / "maitotoxin.sdf"), removeHs=False)
parent = rdMolStandardize.Uncharger().uncharge(rdMolStandardize.FragmentParent(Chem.RemoveHs(parent_3d)))

# The carbon that lost its OH: put the other changes back, then try an OH on each ring CH2
reverted = Chem.RWMol(mol)
for atom in reverted.GetAtoms():
    atom.SetIntProp("original", atom.GetIdx())
reverted.GetBondBetweenAtoms(*alkene).SetBondType(Chem.BondType.SINGLE)
for atom in sorted(benzyl, reverse=True):
    reverted.RemoveAtom(atom)
reverted = reverted.GetMol()
Chem.SanitizeMol(reverted)
found = []
for atom in reverted.GetAtoms():
    if atom.GetSymbol() == "C" and atom.IsInRing() and atom.GetTotalNumHs() == 2:
        trial = Chem.RWMol(reverted)
        trial.AddBond(atom.GetIdx(), trial.AddAtom(Chem.Atom(8)), Chem.BondType.SINGLE)
        trial = trial.GetMol()
        Chem.SanitizeMol(trial)
        if flat(trial) == flat(parent):
            found.append(trial)
if len(found) != 1:
    raise ValueError(f"expected one carbon that gives maitotoxin with an OH, found {len(found)}")
trial = found[0]  # the parent's atoms; all but the OH's O carry their atom number in the derivative ("original")
removed_o = trial.GetNumAtoms() - 1
oh_carbon = trial.GetAtomWithIdx(removed_o).GetNeighbors()[0].GetIntProp("original")
print(f"benzyl {len(benzyl)} atoms, C=C atoms {alkene}, OH removed from atom {oh_carbon}")

# The parent's 3D model, from the 2D drawing: each atom lifted a little off the page (a wedged atom up, a hashed one
# down), then minimised locally (MMFF, distance-dependent dielectric of 80, so hydrogen bonds don't fold it up). Only
# downhill from the drawing, so it stays stretched out.
conformer = parent_3d.GetConformer()
jitter = np.random.default_rng(853).uniform(-0.05, 0.05, parent_3d.GetNumAtoms())  # nothing exactly flat
for i in range(parent_3d.GetNumAtoms()):
    position = conformer.GetAtomPosition(i)
    conformer.SetAtomPosition(i, Point3D(position.x, position.y, jitter[i]))
for bond in parent_3d.GetBonds():
    stereo = bond.GetPropsAsDict().get("_MolFileBondStereo", 0)
    if stereo in (WEDGE, HASH):
        position = conformer.GetAtomPosition(bond.GetEndAtomIdx())
        conformer.SetAtomPosition(bond.GetEndAtomIdx(), Point3D(position.x, position.y, 1.0 if stereo == WEDGE else -1.0))
minimise(parent_3d)
check_stereo(parent_3d, Chem.MolToSmiles(Chem.RemoveHs(Chem.MolFromMolFile(str(STRUCTURES / "website" / "maitotoxin.sdf")))))

# The derivative's model: the parent's model with the changes made to it. Every other atom, hydrogens included,
# keeps its place (so its stereo can't change); only the changed atoms, their neighbours and their hydrogens are
# minimised.
match = parent_3d.GetSubstructMatch(trial, useChirality=True)  # match[i]: the parent's atom for the trial's atom i
if len(match) != trial.GetNumAtoms():
    raise ValueError("the parent's model is not maitotoxin")
in_parent = {atom.GetIntProp("original"): match[i] for i, atom in enumerate(trial.GetAtoms()) if i != removed_o}
benzyl_o = next(n.GetIdx() for n in mol.GetAtomWithIdx(benzyl[0]).GetNeighbors() if n.GetSymbol() == "O")
edited = Chem.RWMol(parent_3d)
for atom in edited.GetAtoms():
    atom.SetIntProp("parent", atom.GetIdx())  # atoms are renumbered as atoms are removed
positions = edited.GetConformer().GetPositions()


def hydrogens(atom):
    return [n.GetIdx() for n in edited.GetAtomWithIdx(atom).GetNeighbors() if n.GetSymbol() == "H"]


gone = [a.GetIdx() for a in edited.GetAtoms() if a.GetSymbol() == "Na"]  # sulfates as the free acid (no Na)
removed_o_parent = match[removed_o]
gone += [removed_o_parent] + hydrogens(removed_o_parent)  # the OH removed
for carbon in (in_parent[alkene[0]], in_parent[alkene[1]]):  # CH2-CH2 to CH=CH: one H off each
    gone.append(hydrogens(carbon)[0])
edited.GetBondBetweenAtoms(in_parent[alkene[0]], in_parent[alkene[1]]).SetBondType(Chem.BondType.DOUBLE)
alcohol_o = in_parent[benzyl_o]
alcohol_h = hydrogens(alcohol_o)[0]
gone.append(alcohol_h)  # the alcohol's H, where the benzyl goes
outward = positions[alcohol_h] - positions[alcohol_o]
outward /= np.linalg.norm(outward)
new = {}  # the benzyl's atoms: derivative atom -> new atom, started in a line outward from the O (then minimised)
for k, m in enumerate(benzyl):
    new[m] = edited.AddAtom(Chem.Atom(6))
    start = positions[alcohol_o] + outward * 1.4 * (k + 1) + np.random.default_rng(k).uniform(-0.3, 0.3, 3)
    edited.GetConformer().SetAtomPosition(new[m], Point3D(*start))
edited.AddBond(alcohol_o, new[benzyl[0]], Chem.BondType.SINGLE)
for bond in mol.GetBonds():
    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    if a in new and b in new:
        edited.AddBond(new[a], new[b], bond.GetBondType())
for atom in edited.GetAtoms():
    if atom.GetSymbol() == "O" and atom.GetFormalCharge() == -1:
        atom.SetFormalCharge(0)
changed_parent = {in_parent[alkene[0]], in_parent[alkene[1]], alcohol_o,
                  next(n.GetIdx() for n in parent_3d.GetAtomWithIdx(removed_o_parent).GetNeighbors() if n.GetSymbol() == "C")}
for atom in sorted(gone, reverse=True):
    edited.RemoveAtom(atom)
model = edited.GetMol()
for atom in model.GetAtoms():
    atom.SetNoImplicit(False)
Chem.SanitizeMol(model)
model = Chem.AddHs(model, addCoords=True)  # the H's the changes need: the sulfates' OH, the carbon that lost its OH
changed = {a.GetIdx() for a in model.GetAtoms()
           if not a.HasProp("parent") and a.GetSymbol() != "H" or a.HasProp("parent") and a.GetIntProp("parent") in changed_parent}
free = changed | {n.GetIdx() for a in changed for n in model.GetAtomWithIdx(a).GetNeighbors()}
free |= {a.GetIdx() for a in model.GetAtoms() if a.GetSymbol() == "H" and (not a.HasProp("parent") or a.GetNeighbors()[0].GetIdx() in free)}
minimise(model, fixed=[a for a in range(model.GetNumAtoms()) if a not in free])
check_stereo(model, Chem.MolToSmiles(mol))
# The derivative's atoms in the model, for the clouds
heavy = Chem.RemoveHs(model)
to_model = heavy.GetSubstructMatch(mol, useChirality=True)  # to_model[i]: the model's atom for the derivative's atom i
xyz = heavy.GetConformer().GetPositions()
parent_xyz = parent_3d.GetConformer().GetPositions()
model = heavy  # drawn without hydrogens


# Turned so the longest direction is across the picture and the next longest up it (3Dmol looks down the z-axis)
middle = xyz.mean(axis=0)
_, _, axes = np.linalg.svd(xyz - middle, full_matrices=False)
if np.linalg.det(axes) < 0:
    axes[2] = -axes[2]  # a rotation, not a mirror image


def turned(points):
    return (np.asarray(points) - middle) @ axes.T


xyz = turned(xyz)
for i, point in enumerate(xyz):
    model.GetConformer().SetAtomPosition(i, Point3D(*point))
removed_at = turned(parent_xyz[removed_o_parent])  # where the OH's O was, in the parent


def listed(point):
    return [round(float(v), 3) for v in point]


clouds = [[listed(xyz[to_model[a]]), BLUE] for a in benzyl] + [[listed(removed_at), RED]]
bonds = [[listed(xyz[to_model[alkene[0]]]), listed(xyz[to_model[alkene[1]]])]]
labels = [  # each change: its label, the point it points at, its colour
    ["Benzyl ether added", listed(xyz[[to_model[a] for a in benzyl]].mean(axis=0)), BLUE],
    ["C=C added", listed(xyz[[to_model[a] for a in alkene]].mean(axis=0)), BLUE],
    ["OH removed", listed(removed_at), RED],
]

# The picture: 3Dmol in headless Chrome, with each label's point on the picture written into the page
page = f"""<html><head><style>body {{ margin: 0; background: {PAGE}; }}
#view {{ position: relative; width: {VIEW[0]}px; height: {VIEW[1]}px; }}</style></head><body>
<div id="view"></div><pre id="points"></pre>
<script src="{JS}"></script>
<script>
const viewer = $3Dmol.createViewer(document.getElementById("view"), {{backgroundColor: "{PAGE}"}});
viewer.addModel({json.dumps(Chem.MolToMolBlock(model))}, "sdf");
viewer.setStyle({{}}, {{stick: {{radius: 0.18}}}});
for (const [[x, y, z], color] of {json.dumps(clouds)}) {{
  viewer.addSphere(Object.assign({{center: {{x: x, y: y, z: z}}, color: color}}, {json.dumps(CLOUD)}));
}}
for (const [start, end] of {json.dumps(bonds)}) {{  // the new C=C: an elongated cloud along it
  viewer.addCylinder(Object.assign({{start: {{x: start[0], y: start[1], z: start[2]}},
                                     end: {{x: end[0], y: end[1], z: end[2]}}, color: "{BLUE}", fromCap: 2, toCap: 2}},
                                    {json.dumps(CLOUD)}));
}}
viewer.zoomTo(); viewer.render();
const points = {json.dumps(labels)}.map(([text, [x, y, z]]) => viewer.modelToScreen({{x: x, y: y, z: z}}));
document.getElementById("points").textContent = JSON.stringify(points.map(p => [p.x, p.y]));
</script></body></html>"""
html = FIGURES / f"{NAME}-3d.html"
html.write_text(page)
shot = FIGURES / f"{NAME}-3d-raw.png"
chrome = [CHROME, "--headless=new", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
          f"--window-size={VIEW[0]},{VIEW[1]}", f"--force-device-scale-factor={SCALE}", "--virtual-time-budget=10000"]
subprocess.run(chrome + [f"--screenshot={shot}", html.as_uri()], check=True, capture_output=True)
dom = subprocess.run(chrome + ["--dump-dom", html.as_uri()], check=True, capture_output=True, text=True).stdout
points = SCALE * np.array(json.loads(dom.split('<pre id="points">')[1].split("</pre>")[0]))
html.unlink()

# Cropped to the molecule, with each label in its place (LABEL_PLACES), joined to its point by a thin line
image = np.asarray(Image.open(shot).convert("RGB"))
shot.unlink()
background = np.array([int(PAGE[i:i + 2], 16) for i in (1, 3, 5)])
drawn = np.argwhere(np.abs(image.astype(int) - background).sum(axis=2) > 30)
(top, left), (bottom, right) = drawn.min(axis=0) - 20, drawn.max(axis=0) + 20
image = image[top:bottom, left:right]
points = points - [left, top]
height, width = image.shape[:2]
figure, ax = plt.subplots(figsize=(width / DPI, height / DPI), dpi=DPI)
figure.patch.set_alpha(0)
ax.imshow(image, extent=(0, width, height, 0))
ax.set_xlim(0, width)
ax.set_ylim(height, 0)
ax.axis("off")
figure.subplots_adjust(0, 0, 1, 1)
for (text, _, color), (x, y) in zip(labels, points):
    across, down = LABEL_PLACES[text]
    ax.annotate(text, (x, y), xytext=(across * width, down * height), ha="center", va="center",
                fontsize=FONT_SIZE, color="#0b0b0b",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=color, linewidth=2),
                arrowprops=dict(arrowstyle="-", color=color, linewidth=1.5, shrinkA=0, shrinkB=4))
path = FIGURES / f"{NAME}-3d.png"
figure.savefig(path, dpi=DPI, transparent=True)
print(f"Saved {path.relative_to(paths.ROOT)}")
