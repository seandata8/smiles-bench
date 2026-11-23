#### Preamble ####
# Purpose: A rotating 3D model of the benchmark's C70 derivative for the website (Discussion, "Trouble with Rings"):
#          C70 with two cyclopropane rings across cage bonds, each carrying two esters
# Author: Sean Murphy
# Date: 23 November 2026
# Pre-requisites: C70's 3D structure (from ChemicalBook, CAS 115383-22-7) as
#                 data/01-raw_data/structures/website/115383-22-7.mol; an internet connection to view the page (it
#                 loads 3Dmol.js from a CDN)
# The two bridged bonds are found by matching C70's cage to the benchmark's molecule by its bonds alone. Each CH2
# starts just outside its bond, which becomes single (both ends sp3); only the CH2s, the four cage atoms they bridge
# and their hydrogens are minimised (UFF: the file's cage bonds are aromatic, which MMFF can't type). Each ester is
# then built on its own (small, so easy to embed), put where one of its bridge's H's was, turned about the new bond to
# keep clear of the rest, and the esters, bridges and bridged cage atoms minimised again, the cage fixed. Drawn as in
# scripts/14-render_3d.py: sticks, the page's background, slow spin, not turned or zoomed by hand. Saves
# website/hardest/_c70-3d.qmd, included in website/index.qmd.
#
#   uv run scripts/16-render_c70_3d.py

import json

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from smiley_bench import paths
from smiley_bench.benchmark import STRUCTURES
from smiley_bench.tables import read_with_canary

NAME = "C70-deriv"
PAGE = "#fbf6e6"  # the page's background (website/styles.css)
SPIN_SPEED = 0.3  # 3Dmol's units; slow, as in scripts/14-render_3d.py
JS = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"
CH2_START = 1.3  # Å out from the middle of the bridged bond, where each CH2 starts
TURNS = 24  # angles tried when turning each ester about its new bond, to keep it clear of the rest
LONGEST_BOND = 1.8  # Å: any bond longer than this after minimising means the model went wrong

c70 = Chem.MolFromMolFile(str(STRUCTURES / "website" / "115383-22-7.mol"))
smiles = read_with_canary(paths.RAW_DATA / "molecule_tests.csv").filter(pl.col("name") == NAME)["smiles_1"][0]
mol = Chem.MolFromSmiles(smiles)

# The bridged bonds: C70's cage in the derivative (matched by bonds alone), then the added atoms bonded to two cage atoms
generic = Chem.AdjustQueryParameters.NoAdjustments()
generic.makeBondsGeneric = True
cage = mol.GetSubstructMatch(Chem.AdjustQueryProperties(c70, generic))  # cage[i]: the derivative's atom for C70's i
if len(cage) != c70.GetNumAtoms():
    raise ValueError("C70's cage was not found in the derivative")
to_c70 = {atom: i for i, atom in enumerate(cage)}
bridged = []  # C70's two atoms for each bridge
bridges = []  # the derivative's bridge carbon for each
for atom in mol.GetAtoms():
    on_cage = [to_c70[n.GetIdx()] for n in atom.GetNeighbors() if n.GetIdx() in to_c70]
    if atom.GetIdx() not in to_c70 and len(on_cage) == 2:
        bridged.append(on_cage)
        bridges.append(atom.GetIdx())
print(f"Bridged cage bonds (C70's atoms): {bridged}")

# C70 with a CH2 across each
cage_xyz = c70.GetConformer().GetPositions()
centre = cage_xyz.mean(axis=0)
edited = Chem.RWMol(c70)
for a, b in bridged:
    edited.GetBondBetweenAtoms(a, b).SetBondType(Chem.BondType.SINGLE)
    ch2 = edited.AddAtom(Chem.Atom(6))
    edited.AddBond(a, ch2, Chem.BondType.SINGLE)
    edited.AddBond(b, ch2, Chem.BondType.SINGLE)
    middle = (cage_xyz[a] + cage_xyz[b]) / 2
    outward = (middle - centre) / np.linalg.norm(middle - centre)
    edited.GetConformer().SetAtomPosition(ch2, Point3D(*(middle + CH2_START * outward)))
model = edited.GetMol()
Chem.SanitizeMol(model)
model = Chem.AddHs(model, addCoords=True)
free = {a for pair in bridged for a in pair} | set(range(c70.GetNumAtoms(), model.GetNumAtoms()))  # CH2s and their H's
field = AllChem.UFFGetMoleculeForceField(model)
for atom in range(model.GetNumAtoms()):
    if atom not in free:
        field.AddFixedPoint(atom)
print(f"C70 + 2 CH2: minimised {len(free)} atoms (0 = converged): {field.Minimize(maxIts=20000)}")
c72 = model


def unit(v):
    return v / np.linalg.norm(v)


def rotation(axis, angle):
    """The matrix that turns points `angle` radians about the unit vector `axis` (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def turn_onto(u, v):
    """The rotation matrix that turns unit vector u onto unit vector v."""
    axis = np.cross(u, v)
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3) if np.dot(u, v) > 0 else rotation(unit(np.cross(u, [1, 0, 0])), np.pi)
    return rotation(unit(axis), np.arccos(np.clip(np.dot(u, v), -1, 1)))


# The derivative's heavy atoms: the cage and bridge carbons from the C70 + 2 CH2 model; each ester built on its own
# (as the ester on a methyl, the methyl standing for its bridge carbon), then put where one of its bridge's H's was
xyz = {cage[i]: p for i, p in enumerate(c72.GetConformer().GetPositions()[: c70.GetNumAtoms()])}
c72_xyz = c72.GetConformer().GetPositions()
for k, bridge in enumerate(bridges):
    ch2 = c70.GetNumAtoms() + k
    xyz[bridge] = c72_xyz[ch2]
    hydrogens = [n.GetIdx() for n in c72.GetAtomWithIdx(ch2).GetNeighbors() if n.GetSymbol() == "H"]
    esters = [n.GetIdx() for n in mol.GetAtomWithIdx(bridge).GetNeighbors() if n.GetIdx() not in to_c70]
    for ester, hydrogen in zip(esters, hydrogens):
        group = {ester}  # the ester's atoms: everything reached from its carbonyl carbon without passing the bridge
        frontier = [ester]
        while frontier:
            for n in mol.GetAtomWithIdx(frontier.pop()).GetNeighbors():
                if n.GetIdx() not in group and n.GetIdx() != bridge:
                    group.add(n.GetIdx())
                    frontier.append(n.GetIdx())
        piece = Chem.RWMol(mol)
        for atom in piece.GetAtoms():
            atom.SetIntProp("original", atom.GetIdx())
        for atom in sorted(set(range(mol.GetNumAtoms())) - group - {bridge}, reverse=True):
            piece.RemoveAtom(atom)
        piece = piece.GetMol()
        for atom in piece.GetAtoms():
            atom.SetNoImplicit(False)
            atom.SetIsAromatic(False)
        Chem.SanitizeMol(piece)
        piece = Chem.AddHs(piece)
        if AllChem.EmbedMolecule(piece, randomSeed=853) != 0:
            raise ValueError(f"the ester on atom {ester} could not be built")
        AllChem.MMFFOptimizeMolecule(piece)
        index = {a.GetIntProp("original"): a.GetIdx() for a in piece.GetAtoms() if a.HasProp("original")}
        heavy = [a.GetIdx() for a in piece.GetAtoms() if a.GetSymbol() != "H"]
        points = piece.GetConformer().GetPositions()
        # Put the methyl carbon on the bridge carbon and the carbonyl carbon along the bridge's C-H bond
        points = points - points[index[bridge]]
        points = points @ turn_onto(unit(points[index[ester]]), unit(c72_xyz[hydrogen] - xyz[bridge])).T
        axis = unit(c72_xyz[hydrogen] - xyz[bridge])
        others = np.array(list(xyz.values()))
        best, clearance = None, -1
        for step in range(TURNS):  # turned about the new bond to the angle that keeps it furthest from the rest
            turned = points @ rotation(axis, 2 * np.pi * step / TURNS).T + xyz[bridge]
            gap = min(np.linalg.norm(others - turned[i], axis=1).min() for i in heavy
                      if i not in (index[bridge], index[ester]))
            if gap > clearance:
                best, clearance = turned, gap
        for original, i in index.items():
            if original != bridge:
                xyz[original] = best[i]
        print(f"Ester on atom {ester}: {len(group)} atoms, clearance {clearance:.1f} Å")

layout = Chem.Conformer(mol.GetNumAtoms())
for atom, position in xyz.items():
    layout.SetAtomPosition(atom, Point3D(*position))
mol.AddConformer(layout, assignId=True)
model = Chem.AddHs(mol, addCoords=True)
added = [a for a in range(mol.GetNumAtoms()) if a not in to_c70]
free = set(added) | {cage[a] for pair in bridged for a in pair}
free |= {a.GetIdx() for a in model.GetAtoms() if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetIdx() in free}
field = AllChem.UFFGetMoleculeForceField(model)
for atom in range(model.GetNumAtoms()):
    if atom not in free:
        field.AddFixedPoint(atom)
print(f"Derivative: minimised {len(free)} atoms (0 = converged): {field.Minimize(maxIts=20000)}")
final = model.GetConformer().GetPositions()
longest = max(np.linalg.norm(final[b.GetBeginAtomIdx()] - final[b.GetEndAtomIdx()]) for b in model.GetBonds())
print(f"Longest bond {longest:.2f} Å")
if longest > LONGEST_BOND:
    raise ValueError(f"a bond of {longest:.2f} Å: the model went wrong")

page = paths.ROOT / "website" / "hardest" / "_c70-3d.qmd"
page.write_text(f"""```{{=html}}
<div id="c70-3d" class="viewer-3d viewer-3d-single"></div>
<script src="{JS}"></script>
<script>
(function () {{
  const viewer = $3Dmol.createViewer(document.getElementById("c70-3d"), {{backgroundColor: "{PAGE}", nomouse: true}});
  viewer.addModel({json.dumps(Chem.MolToMolBlock(Chem.RemoveHs(model), kekulize=False))}, "sdf");
  viewer.setStyle({{}}, {{stick: {{radius: 0.18}}}});
  viewer.zoomTo(); viewer.render(); viewer.spin("y", {SPIN_SPEED});
}})();
</script>
```
""")
print(f"Saved {page.relative_to(paths.ROOT)}")
