import pytest

from dgem_nn.data_processing.drug_profile import (
    build_drugs_names_vs_experiment_map,
    DrugProfile,
)


@pytest.fixture
def drugs_signatures():
    return {
        "aceclofenac_-5_PC3_BRD123": DrugProfile(
            drug_signature=[[1, 1], [1, 1], [1, 1]],
            smiles="CC1C(C(CC(O1)OC2C(OC(CC2O)OC3C(OC(CC3O)OC4CCC5(C(C4)CCC6C5CC(C7(C6(CCC7C8=CC(=O)OC8)O)C)O)C)C)C)O)O",
        ),
        "aceclofenac_-6_MCF7_BRD123": DrugProfile(
            drug_signature=[[0.1, 2], [0.1, 2], [0.1, 2]],
            smiles="CC1C(C(CC(O1)OC2C(OC(CC2O)OC3C(OC(CC3O)OC4CCC5(C(C4)CCC6C5CC(C7(C6(CCC7C8=CC(=O)OC8)O)C)O)C)C)C)O)O",
        ),
        "tamoxifen_-5_PC3_BRD124": DrugProfile(
            drug_signature=[[0.1, -3], [0.1, -3], [0.1, 3]],
            smiles="CC(=O)OC1C(SC2=CC=CC=C2N(C1=O)CCN(C)C)C3=CC=C(C=C3)OC",
        ),
        "tamoxifen_-6_HL60_BRD124": DrugProfile(
            drug_signature=[[0.1, -3], [0.1, -3], [0.1, -3]],
            smiles="CC(=O)OC1C(SC2=CC=CC=C2N(C1=O)CCN(C)C)C3=CC=C(C=C3)OC",
        ),
    }


def test_build_drug_names_vs_experiment_map(drugs_signatures):
    m = build_drugs_names_vs_experiment_map(drugs_signatures=drugs_signatures)
    assert m["BRD124"] == ["tamoxifen_-5_PC3_BRD124", "tamoxifen_-6_HL60_BRD124"]
    assert m["BRD123"] == ["aceclofenac_-5_PC3_BRD123", "aceclofenac_-6_MCF7_BRD123"]
    assert len(m) == 2
