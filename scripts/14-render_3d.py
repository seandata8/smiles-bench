#### Preamble ####
# Purpose: 3D models of a hardest molecule's parent and the benchmark's molecule side by side, rotating together, with
#          the changes highlighted and each SMILES underneath (website Discussion, "Memory Errors")
# Author: Sean Murphy
# Date: 23 November 2026
# Pre-requisites: an internet connection to view the page (it loads 3Dmol.js from a CDN)
# For each molecule in MOLECULES: the parent's model is built by model_3d() (every stereocentre checked against the
# SMILES), and the derivative's is made from it (see make_derivative), so the two line up. Removed atoms are in a red
# cloud on the parent, added atoms in a blue cloud on the derivative, swapped atoms (another element in the same
# place) in a purple cloud on both, and a new bond in an elongated blue cloud. The derivative's
# SMILES is the parent's canonical SMILES with Sean's changes made as text edits (checked to be the benchmark's
# molecule), so the two line up, with the same atoms highlighted. Hydrogens are left out, and the models can't be
# turned or zoomed by hand. Saves website/hardest/_<name>-3d.qmd, included in website/index.qmd.
#
#   uv run scripts/14-render_3d.py

import json

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Geometry import Point3D

from smiley_bench import paths
from smiley_bench.highlight import edited_smiles, smiles_html
from smiley_bench.stereo3d import model_3d
from smiley_bench.tables import read_with_canary

# Each molecule: its parent's name on the page, the parent's SMILES, the edits that make the parent's canonical
# SMILES into the benchmark molecule's, and how the two are matched (compare): BY_ELEMENT, or ANY_ELEMENT for B12,
# where matching rings only to rings breaks up the large ring through the Co
BY_ELEMENT, ANY_ELEMENT = "by element", "any element"
MOLECULES = {
    "vancomycin-deriv": (
        "Vancomycin",
        # From PubChem (CID 14969)
        "C[C@H]1[C@H]([C@@](C[C@@H](O1)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)[C@H]([C@H](C(=O)"
        "N[C@H](C(=O)N[C@H]5C(=O)N[C@@H]7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)[C@H](NC(=O)[C@H]([C@@H](C1=CC(=C(O4)C=C1)Cl)"
        "O)NC7=O)C(=O)O)CC(=O)N)NC(=O)[C@@H](CC(C)C)NC)O)Cl)CO)O)O)(C)N)O",
        [
            ("C(=O)N[C@@H](CC(N)=O)C(=O)", "C(=O)[C@@H](CC(NC)=O)C(=O)"),  # backbone NH removed; N-methyl amide
            ("(N)[C@H](O)[C@H](C)O2", "(N)C[C@H](C)O2"),  # vancosamine 4-OH removed
        ],
        BY_ELEMENT,
    ),
    "calicheamicin-deriv": (
        "Calicheamicin",
        # From PubChem (CID 10953353)
        "CCN[C@H]1CO[C@H](C[C@@H]1OC)O[C@@H]2[C@H]([C@@H]([C@H](O[C@H]2O[C@H]3C#C/C=C\\C#C[C@@]\\4(CC(=O)C(=C3/C4=C\\"
        "CSSSC)NC(=O)OC)O)C)NO[C@H]5C[C@@H]([C@@H]([C@H](O5)C)SC(=O)C6=C(C(=C(C(=C6OC)OC)O[C@H]7[C@@H]([C@@H]([C@H]"
        "([C@@H](O7)C)O)OC)O)I)C)O)O",
        [
            ("CCN[C@H]1", "CN[C@H]1"),  # N-ethyl to N-methyl
            ("C#C[C@]4(O)", "C#C[C@@]4(O)"),  # a centre inverted in the benchmark's molecule (not intended; not shown)
            ("[C@@H](OC)[C@H]5O)", "[C@@H](OC6)[C@H]5O6)"),  # the OMe and OH joined: a methylene acetal
            ("c4OC)[C@@H](C)O3)", "c4OC)CO3)"),  # the thio-sugar's 6-methyl removed
        ],
        BY_ELEMENT,
    ),
    "cyanocobalamin-deriv": (
        "Cyanocobalamin (vitamin B12)",
        # Written as one molecule, as the benchmark's derivative is (Co bonded to the cyanide C and the five N); the
        # same atoms and bonds as PubChem's (CID 166596686, in three pieces)
        "CC1=C2[C@@H](CCC(N)=O)[C@](C)(CC(N)=O)[C@]3(C)[C@H]4[C@H](CC(N)=O)[C@@]5(C)CCC(=O)NC[C@@H](C)OP(=O)([O-])O"
        "[C@H]6[C@@H](O)[C@H](O[C@@H]6CO)n6c[n+](c7cc(C)c(C)cc76)[Co-3]67([C]#N)([N]23)[N+]2=C(C=C3[C@@H](CCC(N)=O)"
        "[C@](C)(CC(N)=O)C1=[N+]36)C(C)(C)[C@H](CCC(N)=O)C2=C(C)C5=[N+]47",
        [
            ("CC1=C2[C@@H](CCC(N)=O)", "CC1=C2[C@@H](CNC(C)=O)"),  # a propionamide to an acetamidomethyl
            ("[C@@]5(C)CCC(=O)NC", "[C@@]5(C)CC(=O)NC"),  # the side chain to the aminopropanol one CH2 shorter
        ],
        ANY_ELEMENT,
    ),
}
RED, BLUE = "#f27373", "#73a8f2"  # as in the 2D drawings and SMILES
PURPLE = "#a983ea"  # a swapped atom; the same colour as mark.swapped in website/styles.css
PAGE = "#fbf6e6"  # the page's background (website/styles.css): see-through clouds need a solid background
CLOUD = {"radius": 1.1, "opacity": 0.6}  # round each changed atom: see-through, so the atoms show
SPIN_SPEED = 0.3  # 3Dmol's units; slow
JS = "https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js"


def compare(parent, mol, matching):
    """Match the two molecules. BY_ELEMENT: RDKit's maximum common substructure (same elements, rings matching rings),
    then outward along chains: an unmatched atom bonded to a matched one is matched to an unmatched atom of the parent bonded
    to its match (the same element if there is a choice). ANY_ELEMENT: the maximum common substructure with atoms of any element matching. A matched atom of another element
    is swapped. The match
    (mol atom -> parent atom), the parent's removed atoms, the derivative's added atoms, its swapped atoms, and its new
    bonds between matched atoms."""
    if matching == ANY_ELEMENT:  # atoms of any element match each other, rings may match chains; no extending
        common = rdFMCS.FindMCS([parent, mol], timeout=60, atomCompare=rdFMCS.AtomCompare.CompareAny)
    else:
        common = rdFMCS.FindMCS([parent, mol], timeout=60, ringMatchesRingOnly=True, completeRingsOnly=True)
    core = Chem.MolFromSmarts(common.smartsString)
    twin = dict(zip(mol.GetSubstructMatch(core), parent.GetSubstructMatch(core)))
    extended = matching == BY_ELEMENT
    while extended:
        extended = False
        for atom in mol.GetAtoms():
            matched = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in twin]
            if atom.GetIdx() in twin or not matched:
                continue
            options = sorted((parent.GetAtomWithIdx(r).GetSymbol() != atom.GetSymbol(), r)
                             for r in range(parent.GetNumAtoms())
                             if r not in twin.values() and parent.GetBondBetweenAtoms(r, twin[matched[0]]))
            if options:
                twin[atom.GetIdx()] = options[0][1]
                extended = True
    removed = [a.GetIdx() for a in parent.GetAtoms() if a.GetIdx() not in twin.values()]
    added = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in twin]
    new_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
                 if b.GetBeginAtomIdx() in twin and b.GetEndAtomIdx() in twin
                 and parent.GetBondBetweenAtoms(twin[b.GetBeginAtomIdx()], twin[b.GetEndAtomIdx()]) is None]
    swapped = [m for m, p in twin.items() if mol.GetAtomWithIdx(m).GetSymbol() != parent.GetAtomWithIdx(p).GetSymbol()]
    return twin, removed, added, swapped, new_bonds


def make_derivative(mol, parent, parent_3d, twin, removed, added, new_bonds):
    """The derivative's model, made from the parent's: the common atoms keep the parent's positions, and each added
    atom takes the position of a removed atom at the same place in the chain (bonded to the parent's atom where the
    added atom's neighbour is; the same element if there is a choice), or else starts next to its placed neighbours.
    Then, where MMFF has parameters, only the changed atoms (and, for a new bond, their neighbours, to make room) and
    their hydrogens are minimised. MMFF has none for Co, so B12's derivative is left as placed. Not an accurate
    structure, just the parent's with the changes shown."""
    conformer = parent_3d.GetConformer()
    layout = Chem.Conformer(mol.GetNumAtoms())
    for m, p in twin.items():
        layout.SetAtomPosition(m, conformer.GetAtomPosition(p))
    stands_for = dict(twin)  # each placed atom: the parent atom whose position it took
    unused = set(removed)
    while len(stands_for) < mol.GetNumAtoms():
        for m in added:
            neighbours = [n.GetIdx() for n in mol.GetAtomWithIdx(m).GetNeighbors() if n.GetIdx() in stands_for]
            if m in stands_for or not neighbours:
                continue
            element = mol.GetAtomWithIdx(m).GetSymbol()
            choices = sorted((parent.GetAtomWithIdx(r).GetSymbol() != element, r)
                             for r in unused if parent.GetBondBetweenAtoms(r, stands_for[neighbours[0]]))
            if choices:
                r = choices[0][1]
                unused.discard(r)
                layout.SetAtomPosition(m, conformer.GetAtomPosition(r))
                stands_for[m] = r
            else:
                centre = sum((layout.GetAtomPosition(n) for n in neighbours), Point3D(0, 0, 0)) / len(neighbours)
                layout.SetAtomPosition(m, centre + Point3D(0.8, 0.8, 0.8))  # a start; the minimisation moves it
                stands_for[m] = None
    mol = Chem.Mol(mol)
    mol.AddConformer(layout, assignId=True)
    mol_3d = Chem.AddHs(mol, addCoords=True)
    if not AllChem.MMFFHasAllMoleculeParams(mol_3d):
        print("Derivative from the parent's model: no force field (MMFF has no parameters), left as placed")
        return mol_3d
    bonded = {a for bond in new_bonds for a in bond}
    changed = set(added) | bonded | {m for m, p in twin.items()  # common atoms bonded to a removed or added atom
                                     if any(n.GetIdx() in removed for n in parent.GetAtomWithIdx(p).GetNeighbors())
                                     or any(n.GetIdx() in added for n in mol.GetAtomWithIdx(m).GetNeighbors())}
    free = changed | {n.GetIdx() for a in bonded for n in mol.GetAtomWithIdx(a).GetNeighbors()}
    free |= {a.GetIdx() for a in mol_3d.GetAtoms() if a.GetSymbol() == "H" and a.GetNeighbors()[0].GetIdx() in free}
    field = AllChem.MMFFGetMoleculeForceField(mol_3d, AllChem.MMFFGetMoleculeProperties(mol_3d))
    for atom in range(mol_3d.GetNumAtoms()):
        if atom not in free:
            field.AddFixedPoint(atom)
    field.Minimize(maxIts=2000)
    print(f"Derivative from the parent's model: {len(free)} atoms minimised, the rest fixed")
    return mol_3d


def page(name, parent_name, parent_smiles, shown_smiles, parent_3d, mol_3d, twin, removed, added, swapped, new_bonds):
    """The two models side by side, rotating, each with its SMILES, as a raw-HTML block for the website."""
    xyz = mol_3d.GetConformer().GetPositions()
    cylinders = [[list(xyz[a]), list(xyz[b])] for a, b in new_bonds]
    parent_xyz = parent_3d.GetConformer().GetPositions()
    # Clouds, drawn as shapes (3Dmol shows those see-through): removed red, added blue, swapped purple on both sides;
    # a new bond has its own cloud, along it
    parent_clouds = ([[list(parent_xyz[a]), RED] for a in removed]
                     + [[list(parent_xyz[twin[a]]), PURPLE] for a in swapped])
    clouds = [[list(xyz[a]), BLUE] for a in added] + [[list(xyz[a]), PURPLE] for a in swapped]
    parent_html = smiles_html(parent_smiles, {a: "removed" for a in removed} | {twin[a]: "swapped" for a in swapped})
    # In the SMILES, a new bond's atoms are marked (with their ring-bond digits, when the bond closes a ring)
    bond_atoms = {a: "added" for bond in new_bonds for a in bond}
    shown_html = smiles_html(shown_smiles, {a: "added" for a in added} | {a: "swapped" for a in swapped} | bond_atoms,
                             ring_digits=bool(new_bonds))
    return f"""```{{=html}}
<div class="molecule-pair">
<figure><div id="{name}-3d-parent" class="viewer-3d"></div><figcaption>{parent_name}</figcaption>
<code class="pair-smiles">{parent_html}</code></figure>
<figure><div id="{name}-3d" class="viewer-3d"></div><figcaption>{name}</figcaption>
<code class="pair-smiles">{shown_html}</code></figure>
</div>
<script src="{JS}"></script>
<script>
(function () {{
  const style = {{stick: {{radius: 0.18}}}};
  function show(id, block, clouds, bonds) {{
    const viewer = $3Dmol.createViewer(document.getElementById(id), {{backgroundColor: "{PAGE}", nomouse: true}});
    viewer.addModel(block, "sdf");
    viewer.setStyle({{}}, style);
    for (const [[x, y, z], color] of clouds) {{
      viewer.addSphere(Object.assign({{center: {{x: x, y: y, z: z}}, color: color}}, {json.dumps(CLOUD)}));
    }}
    for (const [start, end] of bonds) {{  // a new bond: an elongated blue cloud along it (rounded ends)
      viewer.addCylinder(Object.assign({{start: {{x: start[0], y: start[1], z: start[2]}},
                                         end: {{x: end[0], y: end[1], z: end[2]}}, color: "{BLUE}", fromCap: 2, toCap: 2}},
                                        {json.dumps(CLOUD)}));
    }}
    return viewer;
  }}
  const parent = show("{name}-3d-parent", {json.dumps(Chem.MolToMolBlock(Chem.RemoveHs(parent_3d)))},
                      {json.dumps(parent_clouds)}, []);
  const mol = show("{name}-3d", {json.dumps(Chem.MolToMolBlock(Chem.RemoveHs(mol_3d)))},
                   {json.dumps(clouds)}, {json.dumps(cylinders)});
  // Each fills its own box; lined up, they start facing the same way and turn at the same speed
  for (const viewer of [parent, mol]) {{ viewer.zoomTo(); viewer.render(); viewer.spin("y", {SPIN_SPEED}); }}
}})();
</script>
```
"""


molecules = read_with_canary(paths.RAW_DATA / "molecule_tests.csv")
for name, (parent_name, parent_input, edits, matching) in MOLECULES.items():
    print(f"\n== {name}")
    smiles = molecules.filter(pl.col("name") == name)["smiles_1"][0]
    parent_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(parent_input))
    shown_smiles = edited_smiles(parent_smiles, edits, smiles)
    parent, mol = Chem.MolFromSmiles(parent_smiles), Chem.MolFromSmiles(shown_smiles)  # atoms in the SMILES' order
    twin, removed, added, swapped, new_bonds = compare(parent, mol, matching)
    print(f"removed {[parent.GetAtomWithIdx(a).GetSymbol() for a in removed]}, added "
          f"{[mol.GetAtomWithIdx(a).GetSymbol() for a in added]}, swapped "
          f"{[parent.GetAtomWithIdx(twin[a]).GetSymbol() + '>' + mol.GetAtomWithIdx(a).GetSymbol() for a in swapped]}, new bonds "
          f"{[mol.GetAtomWithIdx(a).GetSymbol() + '-' + mol.GetAtomWithIdx(b).GetSymbol() for a, b in new_bonds]}")
    # Heavy atoms keep their SMILES order in the models (model_3d adds the hydrogens after them)
    parent_3d = model_3d(parent_smiles)
    mol_3d = make_derivative(mol, parent, parent_3d, twin, removed, added, new_bonds)
    path = paths.ROOT / "website" / "hardest" / f"_{name}-3d.qmd"
    path.write_text(page(name, parent_name, parent_smiles, shown_smiles, parent_3d, mol_3d, twin, removed, added,
                         swapped, new_bonds))
    print(f"Saved {path.relative_to(paths.ROOT)}")
