#### Preamble ####
# Purpose: Test the 3D stereo tools behind the molecule viewer against RDKit, which handles chiral carbons correctly
# Author: Sean Murphy
# Date: 12 November 2026
# Pre-requisites: none
# Run with `uv run pytest`.

import re

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from smiley_bench.stereo3d import ATOM, TOKEN, labels_from_3d, model_3d
from smiley_bench.viewer import build_model, page_html, ranked_groups

TROGER = "Cc1ccc2c(c1)C[N@@]1C[N@]2Cc2cc(C)ccc21"  # (5S,11S)-Tröger's base


@pytest.mark.parametrize(
    "source",
    ["C[C@H](O)c1ccccc1", "O[C@@H]1CC[C@H](C)CC1N", "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"],
)
def test_labels_from_3d_match_rdkit_on_random_strings(source):
    """Write the molecule 15 random ways, take the labels off, put back labels read from a 3D model, and check RDKit
    reads the same molecule as the source. This tests the neighbour order, including ring closures."""
    model = Chem.AddHs(Chem.MolFromSmiles(source))
    AllChem.EmbedMolecule(model, randomSeed=7)
    Chem.AssignStereochemistryFrom3D(model)
    centres = [a.GetIdx() for a in model.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    expected = Chem.MolToSmiles(Chem.RemoveHs(model))
    for written in Chem.MolToRandomSmilesVect(Chem.RemoveHs(model), 15, randomSeed=3):
        plain = re.sub(r"\[([A-Z][a-z]?)@@?H?\]", lambda m: m.group(1), written)
        labels = labels_from_3d(plain, model, centres)
        match = model.GetSubstructMatch(Chem.MolFromSmiles(plain), useChirality=False)
        rebuilt, k = [], -1
        for token in TOKEN.findall(plain):
            if ATOM.match(token):
                k += 1
                if match[k] in labels:
                    h = "H" if model.GetAtomWithIdx(match[k]).GetTotalNumHs(includeNeighbors=True) == 1 else ""
                    token = f"[{token}{labels[match[k]]}{h}]"
            rebuilt.append(token)
        assert Chem.MolToSmiles(Chem.MolFromSmiles("".join(rebuilt))) == expected, written


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@H](O)c1ccccc1",  # (S)-1-phenylethanol
        "C[C@@H](O)c1ccccc1",  # (R)-1-phenylethanol
        "C[C@@H](C(=O)O)N",  # alanine
        "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O",  # menthol
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",  # glucose
        "C[S@](=O)c1ccccc1",  # a chiral sulfoxide
        "C[C@@]12CC[C@H]3[C@@H](CC=C4C[C@@H](O)CC[C@]34C)[C@@H]1CC[C@@H]2O",  # a steroid ring system (eight centres)
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",  # tryptophan
        "C[C@H]1CCCC[C@@H]1C(=O)O",  # two centres in one ring
        "CC[P@](C)(=O)c1ccccc1",  # a P-stereogenic phosphine oxide
    ],
)
def test_configuration_from_the_model_agrees_with_rdkit(smiles):
    for centre in build_model(smiles, "test")["centres"]:
        assert not centre["tie"], centre["name"]
        assert centre["label"] == centre["rdkit"], centre["name"]


def test_troger_base_both_nitrogens_are_s():
    centres = build_model(TROGER, "Tröger's base")["centres"]
    assert [c["label"] for c in centres] == ["S", "S"]
    assert [c["rdkit"] for c in centres] == [None, None]  # RDKit does not treat them as stereocentres
    assert all(c["fourth"]["kind"] == "lone pair" for c in centres)


def test_troger_priorities_put_the_bridge_first():
    """From each nitrogen: CH2 bridge (C bonded to N, N) > aryl C > ring CH2 (C bonded to C, H, H)."""
    model = model_3d(TROGER)
    bridge = next(
        a.GetIdx()
        for a in model.GetAtoms()
        if a.GetSymbol() == "C" and sum(n.GetSymbol() == "N" for n in a.GetNeighbors()) == 2
    )
    for nitrogen in (a.GetIdx() for a in model.GetAtoms() if a.GetSymbol() == "N"):
        ranked, tie = ranked_groups(model, nitrogen)
        assert ranked[0] == bridge and model.GetAtomWithIdx(ranked[1]).GetIsAromatic() and not tie


def test_impossible_labels_are_refused():
    """Both Tröger nitrogens marked @@ describe a molecule that cannot exist: the bridge couples them."""
    with pytest.raises(ValueError, match="possible combination"):
        model_3d("Cc1ccc2c(c1)C[N@@]1C[N@@]2Cc2cc(C)ccc21")


def test_page_has_the_title_and_data():
    html = page_html(build_model("C[C@H](O)c1ccccc1", "(S)-1-phenylethanol"))
    assert "<title>(S)-1-phenylethanol in 3D</title>" in html
    assert '"label": "S"' in html and "__DATA__" not in html
