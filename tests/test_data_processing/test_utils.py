import pytest

from dgem_nn.data_processing.utils import (
    ranked_predictions_per_contrast,
    encode_smiles_into_fingerprints,
    group_across_dictionaries
)


@pytest.fixture
def dataset_tsv():
    with open("/tmp/dataset.tsv", "w") as f:
        f.write(
            "\t".join(
                ["disease", "disease_contrast", "drug", "drug_signature", "label"]
            )
            + "\n"
        )
        f.write("\t".join(["pain", "HD1", "drugA", "drugA_-5_HL60", "1"]) + "\n")
        f.write("\t".join(["pain", "HD2", "drugA", "drugA_-5_MCF7", "1"]) + "\n")
        f.write("\t".join(["pain", "HD1", "drugB", "drugB_-5_HL60", "1"]) + "\n")
        f.write("\t".join(["pain", "HD2", "drugB", "drugB_-5_MCF7", "1"]) + "\n")
        f.write("\t".join(["pain2", "HD3", "drugC", "drugC_-5_HL60", "1"]) + "\n")
        f.write("\t".join(["pain2", "HD4", "drugC", "drugC_-5_MCF7", "1"]) + "\n")
        f.write("\t".join(["pain4", "HD6", "drugB", "drugB_-5_HL60", "0"]) + "\n")
        f.write("\t".join(["pain4", "HD6", "drugE", "drugE_-5_MCF7", "1"]) + "\n")
        f.write("\t".join(["pain3", "HD5", "drugD", "drugD_-5_HL60", "1"]) + "\n")
        f.write("\t".join(["pain3", "HD5", "drugA", "drugA_-5_MCF7", "0"]) + "\n")
    return "/tmp/dataset.tsv"


def test_ranked_predictions():
    test_set_triples = [
        ("diseaseA", "drugA_HELA_-1_BRD-2_24 h", 0),
        ("diseaseA", "drugC_HELA_-1_BRD-3_24 h", 0),
        ("diseaseA", "drugF_HELA_-1_BRD-4_24 h", 0),
        ("diseaseA", "drugD_HELA_-1_BRD-5_24 h", 0),
        ("diseaseA", "drugG_HELA_-1_BRD-6_24 h", 0),
        ("diseaseB", "drugA_HELA_-2_BRD-2_24 h", 0),
        ("diseaseB", "drugB_HELA_-1_BRD-7_24 h", 0),
        ("diseaseB", "drugC_HELA_-3_BRD-3_24 h", 0),
        ("diseaseB", "drugF_HELA_-5_BRD-4_24 h", 0),
        ("diseaseB", "drugE_HELA_-1_BRD-8_24 h", 0),
        ("diseaseB", "drugE_HELA_-8_BRD-8_24 h", 0),
    ]
    preds = [0.1, 0.5, 0.05, 0.9, 0.2, 0.6, 0.7, 0.96, 0.2, 0.55, 0.7]
    ranked_preds = ranked_predictions_per_contrast(
        test_set_triples=test_set_triples,
        predictions=preds,
        topn=3,
        keep_full_perturbation=False,
    )
    assert ranked_preds["diseaseA"] == [("drugD_BRD-5", 0.9), ("drugC_BRD-3", 0.5), ("drugG_BRD-6", 0.2)]
    assert ranked_preds["diseaseB"] == [
        ("drugC_BRD-3", 0.96),
        ("drugB_BRD-7", 0.7),
        ("drugE_BRD-8", 0.7),
    ]


def test_encode_smiles_into_fingerprints():
    smiles = ["[Li+].[Li+].C(=O)([O-])[O-]", "CC(=O)OC1=CC=CC=C1C(=O)O"]
    fps = encode_smiles_into_fingerprints(smiles, fingeprint_vector_size=20)
    assert fps["[Li+].[Li+].C(=O)([O-])[O-]"].cpu().tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    assert fps["CC(=O)OC1=CC=CC=C1C(=O)O"].cpu().tolist() == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
    ]


def test_group_across_dictionaries():
    d1 = {"a": 1, "b": 2, "c": 3}
    d2 = {"a": 4, "b": 3, "d": 5}
    d3 = {"b": 4, "c": 3, "d": 6}
    d = [d1, d2, d3]
    avg_dict = group_across_dictionaries(dictionaries_list=d)
    assert avg_dict.keys() == {"b"}
    assert avg_dict["b"] == 3.0
