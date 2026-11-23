#### Preamble ####
# Purpose: A rotating 3D model for the website (Discussion, "Trouble with Chirality"): the benchmark's calixapinene,
#          a calix[4]arene with a two-carbon bridge across the bottom and four pinene-tipped arms on the rim
# Author: Sean Murphy
# Date: 23 November 2026
# Pre-requisites: the calix[4]arene from PDB 6SUY (chain A, residue LVT 202) as
#                 data/01-raw_data/structures/website/6SUY-LVT-A.pdb; an internet connection to view the page (it
#                 loads 3Dmol.js from a CDN)
# The core (four aryl rings, four CH2 bridges, four phenolic O's) keeps the crystal structure's positions, so it is
# a real cone. It is matched to calixapinene by its elements and bonds (the PDB has no bond orders). Calixapinene's
# upper-rim arms are cut off (each ring keeps an H there); the two bottom carbons, which join the four O's in pairs,
# start just below their O's, and only they, the O's and their hydrogens are minimised (MMFF), with the stereocentres
# checked against the SMILES. Turned like a bowl (the acetals at the bottom), with its mirror plane as a silver square. Drawn as in scripts/14-render_3d.py. Saves website/hardest/_calixapinene-3d.qmd.
#
#   uv run scripts/17-render_calixapinene_3d.py

import json

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from smiley_bench import paths
from smiley_bench.benchmark import STRUCTURES
from smiley_bench.tables import read_with_canary

NAME = "calixapinene"
PAGE = "#fbf6e6"  # the page's background (website/styles.css)
SPIN_SPEED = 0.3  # 3Dmol's units; slow, as in scripts/14-render_3d.py
PLANE_COLOR, PLANE_OPACITY = "#c0c6cc", 0.35  # the mirror plane: silver, see-through
PLANE_THICKNESS = 0.05  # Å
PLANE_MARGIN = 1.05  # the square's side, as a share of the molecule's width or height (the larger)
ZOOM = 1.2  # after fitting the molecule to the box, so it fills the box better
JS = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"
LONGEST_BOND = 1.8  # Å: any bond longer than this after minimising means the model went wrong
BOTTOM_START = 1.0  # Å below the middle of its two O's, where each bottom carbon starts
TURNS = 24  # angles tried when turning each arm about its new bond, to keep it clear of the rest
# The calix[4]arene core's atoms in the PDB: four aryl rings, four CH2 bridges, four phenolic O's
CORE = ["C2", "C3", "C4", "C5", "C6", "C7", "C10", "C11", "C12", "C13", "C14", "C15", "C25", "C26", "C27", "C28",
        "C29", "C30", "C43", "C44", "C45", "C46", "C47", "C48", "C16", "C17", "C42", "C1", "O5", "O17", "O1", "O21"]

# The core from the crystal structure
pdb = Chem.MolFromPDBFile(str(STRUCTURES / "website" / "6SUY-LVT-A.pdb"), removeHs=False, sanitize=False)
core = Chem.RWMol(pdb)
for atom in sorted((a.GetIdx() for a in pdb.GetAtoms() if a.GetPDBResidueInfo().GetName().strip() not in CORE),
                   reverse=True):
    core.RemoveAtom(atom)
core = core.GetMol()
core_xyz = core.GetConformer().GetPositions()
centre = core_xyz.mean(axis=0)

# Calixapinene, matched to the core by elements and bonds alone
smiles = read_with_canary(paths.RAW_DATA / "molecule_tests.csv").filter(pl.col("name") == NAME)["smiles_1"][0]
mol = Chem.MolFromSmiles(smiles)
skeleton = Chem.RWMol(mol)
for bond in skeleton.GetBonds():
    bond.SetBondType(Chem.BondType.SINGLE)
    bond.SetIsAromatic(False)
for atom in skeleton.GetAtoms():
    atom.SetIsAromatic(False)
generic = Chem.AdjustQueryParameters.NoAdjustments()
generic.makeBondsGeneric = True
match = skeleton.GetSubstructMatch(Chem.AdjustQueryProperties(core, generic))  # match[i]: calixapinene's atom
if len(match) != core.GetNumAtoms():
    raise ValueError("the calix[4]arene core was not found in calixapinene")
to_core = {atom: i for i, atom in enumerate(match)}
oxygens = [atom for atom in match if mol.GetAtomWithIdx(atom).GetSymbol() == "O"]
bottom = sorted({n.GetIdx() for o in oxygens for n in mol.GetAtomWithIdx(o).GetNeighbors() if n.GetIdx() not in to_core})
print(f"Core of {len(match)} atoms; bottom carbons {bottom}")

# The core and the bottom bridge only: the upper-rim arms cut off, each ring keeping an H there
part = Chem.RWMol(mol)
for atom in part.GetAtoms():
    atom.SetIntProp("original", atom.GetIdx())
for atom in sorted(set(range(mol.GetNumAtoms())) - set(match) - set(bottom), reverse=True):
    for neighbour in part.GetAtomWithIdx(atom).GetNeighbors():
        neighbour.SetNoImplicit(False)
    part.RemoveAtom(atom)
part = part.GetMol()
Chem.SanitizeMol(part)
layout = Chem.Conformer(part.GetNumAtoms())
for atom in part.GetAtoms():
    original = atom.GetIntProp("original")
    if original in to_core:
        layout.SetAtomPosition(atom.GetIdx(), Point3D(*core_xyz[to_core[original]]))
    else:  # a bottom carbon: just below the middle of its two O's
        middle = np.mean([core_xyz[to_core[n.GetIntProp("original")]] for n in atom.GetNeighbors()
                          if n.GetIntProp("original") in to_core], axis=0)
        downward = (middle - centre) / np.linalg.norm(middle - centre)
        layout.SetAtomPosition(atom.GetIdx(), Point3D(*(middle + BOTTOM_START * downward)))
part.AddConformer(layout, assignId=True)

model = Chem.AddHs(part, addCoords=True)
free = {a.GetIdx() for a in model.GetAtoms()
        if a.HasProp("original") and (a.GetIntProp("original") in bottom or a.GetIntProp("original") in oxygens)}
free |= {a.GetIdx() for a in model.GetAtoms() if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetIdx() in free}
field = AllChem.MMFFGetMoleculeForceField(model, AllChem.MMFFGetMoleculeProperties(model))
for atom in range(model.GetNumAtoms()):
    if atom not in free:
        field.AddFixedPoint(atom)
print(f"Minimised {len(free)} atoms (0 = converged): {field.Minimize(maxIts=20000)}")
xyz = model.GetConformer().GetPositions()
longest = max(np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()]) for b in model.GetBonds())
check = Chem.Mol(model)
Chem.AssignStereochemistryFrom3D(check)
print(f"Longest bond {longest:.2f} Å")
if longest > LONGEST_BOND:
    raise ValueError(f"a bond of {longest:.2f} Å: the model went wrong")
if Chem.MolToSmiles(Chem.RemoveHs(check)) != Chem.MolToSmiles(part):
    raise ValueError("the model's stereocentres don't match the SMILES")


def rotation(axis, angle):
    """The matrix that turns points `angle` radians about the unit vector `axis` (Rodrigues)."""
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def turn_onto(u, v):
    """The rotation matrix that turns unit vector u onto unit vector v."""
    axis = np.cross(u, v)
    angle = np.arccos(np.clip(np.dot(u, v), -1, 1))
    return rotation(axis / np.linalg.norm(axis), angle) if np.linalg.norm(axis) > 1e-8 else np.eye(3)


# The four upper-rim arms: each built on its own (on a methyl standing for its rim carbon), put where the rim carbon's
# H is, turned about the new bond to keep clear of the rest, then only the arms minimised, the core and bridge fixed
part_xyz = model.GetConformer().GetPositions()
placed = {a.GetIntProp("original"): part_xyz[a.GetIdx()] for a in model.GetAtoms() if a.HasProp("original")}
arms = []
for atom in model.GetAtoms():
    if not atom.HasProp("original") or atom.GetIntProp("original") not in to_core:
        continue
    rim = atom.GetIntProp("original")
    first = [n.GetIdx() for n in mol.GetAtomWithIdx(rim).GetNeighbors() if n.GetIdx() not in to_core]
    if not first or rim in oxygens:
        continue
    hydrogen = next(n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == "H")
    group, frontier = {first[0]}, [first[0]]  # the arm: everything reached from its first atom, away from the rim
    while frontier:
        for n in mol.GetAtomWithIdx(frontier.pop()).GetNeighbors():
            if n.GetIdx() not in group and n.GetIdx() != rim:
                group.add(n.GetIdx())
                frontier.append(n.GetIdx())
    piece = Chem.RWMol(mol)
    for a in piece.GetAtoms():
        a.SetIntProp("original", a.GetIdx())
    for a in sorted(set(range(mol.GetNumAtoms())) - group - {rim}, reverse=True):
        piece.RemoveAtom(a)
    for a in piece.GetAtoms():
        if a.GetIntProp("original") == rim:  # the methyl standing for the rim carbon
            a.SetIsAromatic(False)
            a.SetNoImplicit(False)
            for bond in a.GetBonds():
                bond.SetIsAromatic(False)
                bond.SetBondType(Chem.BondType.SINGLE)
    piece = piece.GetMol()
    Chem.SanitizeMol(piece)
    piece = Chem.AddHs(piece)
    if AllChem.EmbedMolecule(piece, randomSeed=853) != 0:
        raise ValueError(f"the arm on atom {rim} could not be built")
    AllChem.MMFFOptimizeMolecule(piece)
    index = {a.GetIntProp("original"): a.GetIdx() for a in piece.GetAtoms() if a.HasProp("original")}
    points = piece.GetConformer().GetPositions()
    points = points - points[index[rim]]
    along = part_xyz[hydrogen] - part_xyz[atom.GetIdx()]
    along /= np.linalg.norm(along)
    points = points @ turn_onto(points[index[first[0]]] / np.linalg.norm(points[index[first[0]]]), along).T
    others = np.array(list(placed.values()))
    best, clearance = None, -1
    for step in range(TURNS):
        turned = points @ rotation(along, 2 * np.pi * step / TURNS).T + placed[rim]
        gap = min(np.linalg.norm(others - turned[index[a]], axis=1).min() for a in group if a != first[0])
        if gap > clearance:
            best, clearance = turned, gap
    for a in group:
        placed[a] = best[index[a]]
    arms.append(group)
    print(f"Arm on atom {rim}: {len(group)} atoms, clearance {clearance:.1f} Å")

full = Chem.Mol(mol)
for a in full.GetAtoms():
    a.SetIntProp("original", a.GetIdx())
layout = Chem.Conformer(mol.GetNumAtoms())
for a, position in placed.items():
    layout.SetAtomPosition(a, Point3D(*position))
full.AddConformer(layout, assignId=True)
model = Chem.AddHs(full, addCoords=True)
arm_atoms = set().union(*arms)
free = arm_atoms | {a.GetIdx() for a in model.GetAtoms() if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetIdx() in arm_atoms}
field = AllChem.MMFFGetMoleculeForceField(model, AllChem.MMFFGetMoleculeProperties(model))
for atom in range(model.GetNumAtoms()):
    if atom not in free:
        field.AddFixedPoint(atom)
print(f"With the arms: minimised {len(free)} atoms (0 = converged): {field.Minimize(maxIts=20000)}")
xyz = model.GetConformer().GetPositions()
longest = max(np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()]) for b in model.GetBonds())
check = Chem.Mol(model)
Chem.AssignStereochemistryFrom3D(check)
print(f"Longest bond {longest:.2f} Å")
if longest > LONGEST_BOND:
    raise ValueError(f"a bond of {longest:.2f} Å: the model went wrong")
if Chem.MolToSmiles(Chem.RemoveHs(check)) != Chem.MolToSmiles(mol):
    raise ValueError("the model's stereocentres don't match the SMILES")

# Turned like a bowl: the bottom carbons (the acetals) down, the rim up (3Dmol's y-axis is up, and it spins about it)
bottom_atoms = [a.GetIdx() for a in model.GetAtoms() if a.HasProp("original") and a.GetIntProp("original") in bottom]
heavy = [a.GetIdx() for a in model.GetAtoms() if a.GetSymbol() != "H"]
up = xyz[heavy].mean(axis=0) - xyz[bottom_atoms].mean(axis=0)
up /= np.linalg.norm(up)
axis = np.cross(up, [0, 1, 0])  # turn `up` onto the y-axis about this axis (Rodrigues)
sin, cos = np.linalg.norm(axis), np.dot(up, [0, 1, 0])
k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]) / sin
turn = np.eye(3) + sin * k + (1 - cos) * k @ k
for i, point in enumerate((xyz - xyz.mean(axis=0)) @ turn.T):
    model.GetConformer().SetAtomPosition(i, Point3D(*point))

# The mirror plane: through both bottom carbons and up the bowl's axis (it swaps the rings each bottom carbon joins,
# and with them the S and R pinenes). The model is turned about the (vertical) axis until the plane faces the z-axis,
# so the plane can be drawn as a thin silver square (3Dmol draws boxes along its axes), big enough to cover the molecule
xyz = model.GetConformer().GetPositions()
normal = np.cross(xyz[bottom[1]] - xyz[bottom[0]], [0, 1, 0])  # horizontal, as the plane is vertical
angle = np.arctan2(normal[0], normal[2])
facing = rotation(np.array([0.0, 1.0, 0.0]), -angle)  # turns the normal onto the z-axis
xyz = xyz @ facing.T
for i, point in enumerate(xyz):
    model.GetConformer().SetAtomPosition(i, Point3D(*point))

# Made exactly mirror-symmetric: each atom paired with its mirror partner (the symmetry of the molecule's bonds that best
# matches a reflection in the plane, judged on the core); the core and bottom carbons averaged with their partners'
# reflections, and the arms on one side reflected onto the other (a reflected S pinene is an R pinene). The hydrogens are
# then placed again, and the stereocentres checked.
plane_z = float(xyz[bottom[0]][2])


def reflected(points):
    return points * [1, 1, -1] + [0, 0, 2 * plane_z]


fixed_part = list(match) + bottom  # the core and bottom carbons
partner = min(mol.GetSubstructMatches(mol, uniquify=False, useChirality=False, maxMatches=100000),
              key=lambda p: sum(np.linalg.norm(reflected(xyz[i]) - xyz[p[i]]) for i in fixed_part))  # partner[i]
heavy_count = mol.GetNumAtoms()
mirrored = xyz[:heavy_count].copy()
for i in fixed_part:
    mirrored[i] = (xyz[i] + reflected(xyz[partner[i]])) / 2
kept = [arm for arm in arms if np.mean([xyz[i][2] for i in arm]) > plane_z]  # the arms on one side
for arm in kept:
    for i in arm:
        mirrored[partner[i]] = reflected(xyz[i])
print(f"Mirrored {len(kept)} arms onto the other side")
symmetric = Chem.Mol(mol)
layout = Chem.Conformer(heavy_count)
for i, point in enumerate(mirrored):
    layout.SetAtomPosition(i, Point3D(*point))
symmetric.AddConformer(layout, assignId=True)
model = Chem.AddHs(symmetric, addCoords=True)
xyz = model.GetConformer().GetPositions()
longest = max(np.linalg.norm(xyz[b.GetBeginAtomIdx()] - xyz[b.GetEndAtomIdx()]) for b in model.GetBonds())
check = Chem.Mol(model)
Chem.AssignStereochemistryFrom3D(check)
print(f"Symmetric model: longest bond {longest:.2f} Å")
if longest > LONGEST_BOND:
    raise ValueError(f"a bond of {longest:.2f} Å: the symmetric model went wrong")
if Chem.MolToSmiles(Chem.RemoveHs(check)) != Chem.MolToSmiles(mol):
    raise ValueError("the symmetric model's stereocentres don't match the SMILES")
heavy_xyz = xyz[[a.GetIdx() for a in model.GetAtoms() if a.GetSymbol() != "H"]]
low, high = heavy_xyz.min(axis=0), heavy_xyz.max(axis=0)
side = PLANE_MARGIN * max(high[0] - low[0], high[1] - low[1])
square = {"x": float((low[0] + high[0]) / 2), "y": float((low[1] + high[1]) / 2), "z": float(xyz[bottom[0]][2])}

page = paths.ROOT / "website" / "hardest" / f"_{NAME}-3d.qmd"
page.write_text(f"""```{{=html}}
<div id="{NAME}-3d" class="viewer-3d viewer-3d-single"></div>
<script src="{JS}"></script>
<script>
(function () {{
  const viewer = $3Dmol.createViewer(document.getElementById("{NAME}-3d"), {{backgroundColor: "{PAGE}", nomouse: true}});
  viewer.addModel({json.dumps(Chem.MolToMolBlock(Chem.RemoveHs(model)))}, "sdf");
  viewer.setStyle({{}}, {{stick: {{radius: 0.18}}}});
  viewer.addBox({{center: {json.dumps(square)}, dimensions: {{w: {side:.2f}, h: {side:.2f}, d: {PLANE_THICKNESS}}},
                 color: "{PLANE_COLOR}", opacity: {PLANE_OPACITY}}});  // the mirror plane
  viewer.zoomTo(); viewer.zoom({ZOOM}); viewer.render(); viewer.spin("y", {SPIN_SPEED});
}})();
</script>
```
""")
print(f"Saved {page.relative_to(paths.ROOT)}")
