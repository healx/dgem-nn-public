import os

import pytest

from dgem_nn.data_processing.disease_profile import load_omicsoft_diseases_signatures


@pytest.fixture
def omicsoft_signatures_dir():
    os.makedirs("/tmp/diseases_signatures", exist_ok=True)
    with open("/tmp/diseases_signatures/HumanDisease_B37_22.tsv", "w") as f:
        f.write(
            ",ComparisonIndex,GeneIndex,Name,Log2FoldChange,PValue,AdjustedPValue,NumeratorValue,DenominatorValue\n"
            "22,22,0,DDX11L1,0.1561,4.973e-001,1.000e+000,19.1725,13.6820\n"
            "16740,22,1,WASH7P,-0.1415,1.235e-001,9.000e-001,1885.3970,2086.9360\n"
            "71852,22,3,FAM138F,-0.2373,2.776e-001,1.000e+000,0.8246,1.5116"
        )

    with open("/tmp/diseases_signatures/HumanDisease_B37_23.tsv", "w") as f:
        f.write(
            ",ComparisonIndex,GeneIndex,Name,Log2FoldChange,PValue,AdjustedPValue,NumeratorValue,DenominatorValue\n"
            "22,22,0,DDX11L1,0.1561,4.973e-001,1.000e+000,19.1725,13.6820\n"
            "16740,22,1,WASH7P,-0.1415,1.235e-001,9.000e-001,1885.3970,2086.9360\n"
            "66159,22,2,MI0006363,0.2,1.0,1.0,0,0\n"
            "71852,22,3,FAM138F,-0.2373,2.776e-001,1.000e+000,0.8246,1.5116"
        )
    return "/tmp/diseases_signatures"


def test_load_omicsoft_disease_signatures(omicsoft_signatures_dir):
    diseases_signatures = load_omicsoft_diseases_signatures(
        diseases_signatures_path=omicsoft_signatures_dir,
        max_nr_of_genes_in_disease_contrast=4,
    )
    assert len(diseases_signatures) == 2
    assert diseases_signatures["HumanDisease_B37_22"].tolist() == [
        [1.0, 0.1561],
        [0.9, -0.1415],
        [1.0, 0.0],
        [1.0, -0.2373],
    ]
    assert diseases_signatures["HumanDisease_B37_23"].tolist() == [
        [1.0, 0.1561],
        [0.9, -0.1415],
        [1.0, 0.2],
        [1.0, -0.2373],
    ]
