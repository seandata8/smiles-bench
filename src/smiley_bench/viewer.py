#### Preamble ####
# Purpose: Make an interactive 3D viewer page for any molecule, to check its stereocentres: the model turns so a
#          centre's lowest-priority group points away, and the page reads R or S from the geometry
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: uv add rdkit numpy
# Used by scripts/view_molecule.py. The page itself is viewer_template.html, next to this file.
#
# Centres are the atoms given a stereo mark (@ or @@) in the SMILES, including marks RDKit would remove. Their
# neighbours are ranked by the first CIP rule only (atomic numbers), using the CIP tree: each branch is explored level
# by level outwards from the centre, comparing the substituents of the highest-ranked atoms first, with duplicate atoms
# for double bonds (Kekulé structure) and for atoms reached again round a ring; an implicit H or lone pair ranks
# lowest. That settles most drug-like centres but not ones that need the later rules (isotopes, stereo), which are
# reported as ties. Where
# RDKit labels a centre itself, its label is shown beside the one read from the model, as a check.

import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler, rdMolDescriptors

from smiley_bench.stereo3d import implicit_direction, model_3d, written_labels

TEMPLATE = Path(__file__).with_name("viewer_template.html")
CIP_SPHERES = 6  # how many levels out the ranking looks before calling a tie


def children(kekule, node):
    """The substituents of one node of the CIP tree, as (atom, path, is_duplicate) nodes. `path` holds the atoms from
    the centre to this node. A double or triple bond adds duplicate atoms; an atom already on the path (reached again
    round a ring) becomes a duplicate. Duplicates have no substituents of their own (only phantoms, atomic number 0)."""
    atom, path, is_duplicate = node
    if is_duplicate:
        return []
    parent = path[-2]
    found = []
    for bond in kekule.GetAtomWithIdx(atom).GetBonds():
        other = bond.GetOtherAtomIdx(atom)
        if other != parent:
            found.append((other, path + (other,), other in path))
        found += [(other, path + (other,), True)] * (int(bond.GetBondTypeAsDouble()) - 1)
    return found


def atomic_number(kekule, node):
    return kekule.GetAtomWithIdx(node[0]).GetAtomicNum()


def sibling_key(kekule, node, depth=3):
    """A quick key to put sibling nodes in order: atomic number, then their substituents' keys, a few levels deep."""
    if depth == 0:
        return (atomic_number(kekule, node),)
    below = sorted((sibling_key(kekule, child, depth - 1) for child in children(kekule, node)), reverse=True)
    return (atomic_number(kekule, node), tuple(below))


def branch_key(kekule, centre, start):
    """The branch that leaves `centre` through `start`, compared the CIP way (first rule: atomic numbers): level by
    level outwards, and within a level, the substituent sets of each node in turn, highest-ranked node first. Each set
    is sorted highest first and padded with phantom atoms (0) to three. Comparing two keys compares the branches."""
    root = (start, (centre, start), start == centre)
    levels = [((atomic_number(kekule, root),),)]
    current = [root]
    for _ in range(CIP_SPHERES):
        groups, next_level = [], []
        for node in current:
            kids = sorted(children(kekule, node), key=lambda child: sibling_key(kekule, child), reverse=True)
            numbers = [atomic_number(kekule, kid) for kid in kids]
            groups.append(tuple(numbers + [0] * (3 - len(numbers))))
            next_level += kids
        levels.append(tuple(groups))
        current = next_level
    return tuple(levels)


def ranked_groups(model, centre):
    """The centre's neighbours from highest to lowest CIP priority (first rule), and whether any two tie."""
    kekule = Chem.Mol(model)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    neighbours = [a.GetIdx() for a in model.GetAtomWithIdx(centre).GetNeighbors()]
    keys = {n: branch_key(kekule, centre, n) for n in neighbours}
    ordered = sorted(neighbours, key=lambda n: keys[n], reverse=True)
    tie = len(set(keys.values())) < len(keys)
    return ordered, tie


def configuration(positions, centre, groups, fourth):
    """R or S, from the positions of the three highest-priority groups and the lowest one: seen with the lowest
    pointing away, 1 -> 2 -> 3 anticlockwise is S."""
    c = positions[centre]
    v1, v2, v3 = (positions[g] - c for g in groups)
    away = np.asarray(fourth) - c
    return "S" if np.dot(np.cross(v2 - v1, v3 - v1), -away) > 0 else "R"


def rdkit_labels(smiles):
    """RDKit's own CIP labels {atom: "R" or "S"}, for the centres RDKit recognises."""
    mol = Chem.MolFromSmiles(smiles)
    rdCIPLabeler.AssignCIPLabels(mol)
    return {a.GetIdx(): a.GetProp("_CIPCode") for a in mol.GetAtoms() if a.HasProp("_CIPCode")}


def atom_name(model, index):
    """E.g. "N9": the element and the atom's position in the SMILES (counting from 1)."""
    return f"{model.GetAtomWithIdx(index).GetSymbol()}{index + 1}"


def build_model(smiles, name):
    """Everything the page needs: atoms, bonds (Kekulé), and for each marked centre its ranked groups, its lowest
    group (an H, an atom or a lone pair), the R/S read from the model, and RDKit's label if it has one."""
    model = model_3d(smiles)
    positions = model.GetConformer().GetPositions()
    positions = positions - positions.mean(axis=0)
    kekule = Chem.Mol(model)
    Chem.Kekulize(kekule, clearAromaticFlags=True)
    rdkit = rdkit_labels(smiles)
    centres = []
    for centre in sorted(written_labels(smiles)):
        ranked, tie = ranked_groups(model, centre)
        if len(ranked) == 4:
            groups, fourth_atom = ranked[:3], ranked[3]
            fourth = positions[fourth_atom]
            kind = "H" if model.GetAtomWithIdx(fourth_atom).GetSymbol() == "H" else "atom"
        else:
            groups, fourth_atom, kind = ranked, None, "lone pair"
            fourth = implicit_direction(model, centre, positions)
        centres.append(
            {
                "atom": centre,
                "name": atom_name(model, centre),
                "groups": groups,
                "group_names": [atom_name(model, g) for g in groups],
                "fourth": {"kind": kind, "atom": fourth_atom, "xyz": [round(float(v), 3) for v in fourth]},
                "label": configuration(positions, centre, groups, fourth),
                "rdkit": rdkit.get(centre),
                "tie": tie,
            }
        )
    return {
        "name": name,
        "smiles": smiles,
        "formula": rdMolDescriptors.CalcMolFormula(model),
        "atoms": [
            {"el": a.GetSymbol(), "xyz": [round(float(v), 3) for v in positions[a.GetIdx()]]} for a in model.GetAtoms()
        ],
        "bonds": [[b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble())] for b in kekule.GetBonds()],
        "centres": centres,
    }


def page_html(data):
    """The viewer page, with the model's data written into it."""
    html = TEMPLATE.read_text()
    return html.replace("__TITLE__", data["name"]).replace("__DATA__", json.dumps(data))
