#### Preamble ####
# Purpose: A benchmark molecule's SMILES written as its parent's with the changes made as text edits, and SMILES as
#          HTML with some atoms highlighted, for the website's side-by-side molecules (scripts/12 and 14)
# Author: Sean Murphy
# Date: 23 November 2026

import re

from rdkit import Chem

# An atom in a SMILES: in brackets, a two-letter element, or a one-letter one (aromatic in lower case)
ATOM = re.compile(r"\[[^\]]*\]|Br|Cl|[BCNOPSFIbcnops]")
RING_DIGITS = re.compile(r"(?:\d|%\d\d)*")  # the ring-bond digits right after an atom, if any


def edited_smiles(parent_smiles, edits, smiles):
    """The parent's SMILES with `edits` made, checked to be the same molecule as `smiles`."""
    edited = parent_smiles
    for old, new in edits:
        if edited.count(old) != 1:
            raise ValueError(f"{old} is not in the parent's SMILES exactly once")
        edited = edited.replace(old, new)
    if Chem.MolToSmiles(Chem.MolFromSmiles(edited)) != Chem.MolToSmiles(Chem.MolFromSmiles(smiles)):
        raise ValueError(f"the edited SMILES {edited} is not the molecule {smiles}")
    return edited


def smiles_html(smiles, marks, ring_digits=False):
    """The SMILES as HTML, each atom in `marks` ({atom: kind}, the atom by its order in the SMILES) in
    <mark class="kind">; with ring_digits, the ring-bond digits written after a marked atom are marked with it."""
    pieces, end = [], 0
    for number, found in enumerate(ATOM.finditer(smiles)):
        pieces.append(smiles[end:found.start()])
        text, end = found.group(), found.end()
        if number in marks:
            if ring_digits:
                digits = RING_DIGITS.match(smiles, end)
                text, end = text + digits.group(), digits.end()
            text = f'<mark class="{marks[number]}">{text}</mark>'
        pieces.append(text)
    pieces.append(smiles[end:])
    return "".join(pieces)
