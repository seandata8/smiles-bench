#### Preamble ####
# Purpose: Build a 3D model of a molecule that matches the stereo marks written in its SMILES, including marks RDKit
#          removes (e.g. on the bridgehead nitrogens of Tröger's base), and read @/@@ labels off a 3D model
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: uv add rdkit numpy
# Used by the molecule viewer (src/smiley_bench/viewer.py).
#
# The @/@@ label of an atom depends on the order of its neighbours in the SMILES string (OpenSMILES): the previous
# atom, then the implicit H or lone pair, then ring-closure partners in the order their digits follow the atom, then
# branches and the next atom. RDKit does not keep ring-closure bonds in that order, so the order is read from the
# string itself. Checked against RDKit on random SMILES strings of chiral carbons (tests/test_viewer.py).

import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

TOKEN = re.compile(r"\[[^\]]+\]|Br|Cl|%\d\d|[BCNOPSFIbcnops]|\d|[-=#$:/\\.()]")
ATOM = re.compile(r"\[[^\]]+\]|Br|Cl|[BCNOPSFIbcnops]$")


def smiles_neighbours(smiles):
    """For each atom, in the order it appears in the string: its neighbours in SMILES order, and whether it has a
    previous atom (the first atom of a string, or after a dot, has none)."""
    neighbours, has_previous = [], []
    previous, stack, open_rings = None, [], {}
    for token in TOKEN.findall(smiles):
        if ATOM.match(token):
            k = len(neighbours)
            neighbours.append([])
            has_previous.append(previous is not None)
            if previous is not None:
                neighbours[k].append(previous)
                neighbours[previous].append(k)
            previous = k
        elif token == "(":
            stack.append(previous)
        elif token == ")":
            previous = stack.pop()
        elif token == ".":
            previous = None
        elif token[0] in "0123456789%":
            if token in open_rings:
                j, slot = open_rings.pop(token)
                neighbours[j][slot] = previous
                neighbours[previous].append(j)
            else:
                open_rings[token] = (previous, len(neighbours[previous]))
                neighbours[previous].append(None)  # filled in when the ring closes
    return neighbours, has_previous


def written_labels(smiles):
    """{atom index (string order): "@" or "@@"} for each atom given a stereo mark in the string."""
    labels = {}
    atom_index = -1
    for token in TOKEN.findall(smiles):
        if ATOM.match(token):
            atom_index += 1
            mark = re.search(r"@@?", token)
            if mark and not re.search(r"@(TH|AL|SP|TB|OH)", token):
                labels[atom_index] = mark.group()
    return labels


def unlabelled(smiles):
    """The SMILES with every @ and @@ removed (and brackets kept only where still needed by RDKit's parser)."""
    return re.sub(r"@@?", "", smiles)


def signed_volume(points):
    p0, p1, p2, p3 = (np.asarray(p) for p in points)
    return float(np.dot(p1 - p0, np.cross(p2 - p0, p3 - p0)))


def implicit_direction(mol, atom, positions):
    """Where an atom's implicit neighbour points: its explicit H in the model, or else its lone pair (opposite the
    three bonds)."""
    hydrogens = [a.GetIdx() for a in mol.GetAtomWithIdx(atom).GetNeighbors() if a.GetSymbol() == "H"]
    if hydrogens:
        return positions[hydrogens[0]]
    heavy = [a.GetIdx() for a in mol.GetAtomWithIdx(atom).GetNeighbors()]
    unit = sum((positions[i] - positions[atom]) / np.linalg.norm(positions[i] - positions[atom]) for i in heavy)
    return positions[atom] - unit / np.linalg.norm(unit)


def labels_from_3d(smiles, model, atoms, match=None):
    """{model atom: "@" or "@@"}: how each listed atom must be labelled in `smiles` to match the 3D `model` (a
    molecule with explicit hydrogens and coordinates). `smiles` may be labelled or not; only its order is used.
    `match` gives the model atom for each atom of the string, in string order. Leave it out only when the string was
    not used to build the model: it is then found by substructure search, which may swap symmetric atoms."""
    if match is None:
        match = model.GetSubstructMatch(Chem.MolFromSmiles(unlabelled(smiles)), useChirality=False)
    in_string = {model_atom: i for i, model_atom in enumerate(match)}
    neighbours, has_previous = smiles_neighbours(smiles)
    positions = model.GetConformer().GetPositions()
    labels = {}
    for atom in atoms:
        i = in_string[atom]
        heavy = [positions[match[n]] for n in neighbours[i]]
        if len(heavy) == 4:
            ordered = heavy
        else:
            implicit = implicit_direction(model, atom, positions)
            ordered = [heavy[0], implicit, *heavy[1:]] if has_previous[i] else [implicit, *heavy]
        labels[atom] = "@" if signed_volume(ordered) < 0 else "@@"
    return labels


def model_3d(smiles, seed=853, tries=20):
    """A force-field-minimised 3D model (explicit hydrogens) that matches every stereo mark written in `smiles`.
    RDKit keeps most marks while building the model, but not all (e.g. aryl nitrogens), and those come out either
    way. So the model is built with up to `tries` seeds, each also checked as its mirror image (enough when every
    mark is one RDKit removes), and the first that matches every written mark is kept. Several tries are needed when
    a molecule has both kinds, e.g. a carbon centre and the nitrogens of Tröger's base. An error if none matches."""
    wanted = written_labels(smiles)
    # A model built from this string has its heavy atoms in string order.
    in_order = list(range(Chem.MolFromSmiles(unlabelled(smiles)).GetNumAtoms()))

    def matches(candidate):
        return labels_from_3d(smiles, candidate, list(wanted), match=in_order) == wanted

    for attempt in range(tries):
        model = Chem.AddHs(Chem.MolFromSmiles(smiles))
        if AllChem.EmbedMolecule(model, randomSeed=seed + attempt) != 0:
            continue
        AllChem.MMFFOptimizeMolecule(model, maxIters=2000)
        if not wanted or matches(model):
            return model
        mirror = Chem.Mol(model)
        conformer = mirror.GetConformer()
        for i in range(mirror.GetNumAtoms()):
            p = conformer.GetAtomPosition(i)
            conformer.SetAtomPosition(i, (-p.x, p.y, p.z))
        if matches(mirror):
            return mirror
    raise ValueError(f"No 3D model of {smiles} matches all its stereo marks: are they a possible combination?")
