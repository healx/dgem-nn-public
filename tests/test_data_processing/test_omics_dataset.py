import numpy
import pytest
import torch
from torch.utils.data import DataLoader

from dgem_nn.data_processing.drug_profile import DrugProfile
from dgem_nn.data_processing.omics_dataset import (
    OmicsDataset,
    get_weighted_sampler,
    load_drug_profiles_as_tensors,
)
from dgem_nn.data_processing.utils import (
    encode_smiles_into_fingerprints,
)


@pytest.fixture
def contrast_drug_pairs():
    return [
        ("HumanDisease.B37_9866", "digoxin_-5_PC3", 0),
        ("HumanDisease.B37_2591", "diltiazem_-6_MCF7", 0),
        ("HumanDisease.B37_2591", "diltiazem_-8_HL60", 1),
    ]


@pytest.fixture
def drugs_signatures():
    return {
        "digoxin_-5_PC3": DrugProfile(
            drug_signature=[1, 1, 1],
            smiles="CC1C(C(CC(O1)OC2C(OC(CC2O)OC3C(OC(CC3O)OC4CCC5(C(C4)CCC6C5CC(C7(C6(CCC7C8=CC(=O)OC8)O)C)O)C)C)C)O)O",
        ),
        "diltiazem_-6_MCF7": DrugProfile(
            drug_signature=[2, 2, 2],
            smiles="CC(=O)OC1C(SC2=CC=CC=C2N(C1=O)CCN(C)C)C3=CC=C(C=C3)OC",
        ),
        "diltiazem_-8_HL60": DrugProfile(
            drug_signature=[3, 3, 3],
            smiles="CC(=O)OC1C(SC2=CC=CC=C2N(C1=O)CCN(C)C)C3=CC=C(C=C3)OC",
        ),
    }


@pytest.fixture
def omics_dataset(contrast_drug_pairs, drugs_signatures):
    diseases_signatures = {
        "HumanDisease.B37_9866": numpy.array([4, 4, 0]),
        "HumanDisease.B37_2591": numpy.array([0, 5, 5]),
    }

    smiles = list(
        set([drug_profile.smiles for drug_profile in drugs_signatures.values()])
    )
    compounds_fingerprints_dict = encode_smiles_into_fingerprints(
        smiles=smiles, fingeprint_vector_size=5
    )
    omics_dataset = OmicsDataset(
        drugs_signatures=drugs_signatures,
        diseases_signatures=diseases_signatures,
        contrast_drug_label_triples=contrast_drug_pairs,
    )
    return omics_dataset


def test_omics_dataset(contrast_drug_pairs, omics_dataset):
    assert len(omics_dataset) == len(contrast_drug_pairs)
    loader = DataLoader(dataset=omics_dataset, shuffle=False, batch_size=1)
    first_instance = [element for element in loader][0]
    assert first_instance["disease_contrast"].tolist()[0] == [4, 4, 0]
    assert first_instance["drug_signature"].tolist()[0] == [1, 1, 1]
    assert first_instance["label"] == 0


def test_get_weighted_sampler(omics_dataset):
    # below is a dummy way to increase the dset size
    dset = torch.utils.data.ConcatDataset(
        [
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
            omics_dataset,
        ]
    )
    sampler = get_weighted_sampler(labels=[instance["label"] for instance in dset])
    train_dataloader = DataLoader(
        dataset=dset,
        sampler=sampler,
        batch_size=8,
        num_workers=0,
    )

    labels = [label for batch in train_dataloader for label in batch["label"]]
    assert len(labels) == 72
    assert labels.count(1) == pytest.approx(36, abs=10)


def test_load_drug_profiles_as_tensors(drugs_signatures):
    smiles = list(
        set([drug_profile.smiles for drug_profile in drugs_signatures.values()])
    )
    compounds_fingerprints_dict = encode_smiles_into_fingerprints(
        smiles=smiles, fingeprint_vector_size=5
    )
    tensorized_profiles = load_drug_profiles_as_tensors(
        drug_profiles=drugs_signatures,
    )
    assert tensorized_profiles["digoxin_-5_PC3"].cpu().tolist() == [1.0, 1.0, 1.0]
    assert tensorized_profiles["diltiazem_-6_MCF7"].cpu().tolist() == [2.0, 2.0, 2.0]
    assert tensorized_profiles["diltiazem_-8_HL60"].cpu().tolist() == [3.0, 3.0, 3.0]
