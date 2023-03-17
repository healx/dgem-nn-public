import logging
from collections import Counter
from typing import List

import numpy

from dgem_nn.data_processing.utils import load_as_tensors

logging.getLogger("deepchem").setLevel(logging.WARNING)
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

logger = logging.getLogger(__name__)


def load_drug_profiles_as_tensors(
        drug_profiles: dict,
):
    """
    Load drug profiles and their SMILES fingerprint representations as Torch tensors.
    This is used in order to preload the tensors in GPU and save time during training or
     inference.

    """
    tensorized_drug_profiles = {}
    drug_profiles_tuples = list(drug_profiles.items())
    drugs_signatures = numpy.array(
        [drug_profile.drug_signature for _, drug_profile in drug_profiles_tuples]
    )
    tensorized_drugs_signatures = torch.Tensor(drugs_signatures)
    for i, (drug_perturbation_name, drug_profile) in enumerate(drug_profiles_tuples):
        tensorized_signature = tensorized_drugs_signatures[i]
        tensorized_drug_profiles[drug_perturbation_name] = tensorized_signature
    return tensorized_drug_profiles


class OmicsDataset(Dataset):
    def __init__(
            self,
            drugs_signatures: dict,
            diseases_signatures: dict,
            contrast_drug_label_triples: List,
    ):
        """
        The torch dataset class used to train a NN.
        :param drugs_signatures: dictionary of DrugProfile instances, each element has
        a key of the form drugname_cellline_dosage_broadname and contains as value a
        DrugProfile with SMILES and drug vector (=omics profile)
        :param diseases_signatures: dictionary of contrast ids and their omics profile
        :param contrast_drug_label_triples: List of triples (disease contrast, drug
        perturbation, label)
        # :param cell_lines_encoder: one hot encoder of cell lines
        # :param dosages_encoder: one hot encoder of dosages
        # :param compounds_fingerprints_dict: dictionary with Morgan fingerprint vectors
        for each smiles. This is used to create a structural representation of SMILES
        strings.

        """
        self.contrast_drug_label_triples = contrast_drug_label_triples
        self._validate_contrast_drug_triples_format(self.contrast_drug_label_triples)
        self.labels = [triple[2] for triple in contrast_drug_label_triples]

        self.drugs_signatures = load_drug_profiles_as_tensors(
            drug_profiles=drugs_signatures
        )
        self.diseases_signatures = load_as_tensors(diseases_signatures)

    def __getitem__(self, index):
        (
            disease_contrast_id,
            drug_signature_id,
            label,
        ) = self.contrast_drug_label_triples[index]
        disease_contrast = self.diseases_signatures[disease_contrast_id]
        tensorized_drug_signature = self.drugs_signatures[drug_signature_id]
        return {
            "disease_contrast": disease_contrast,
            "drug_signature": tensorized_drug_signature,
            "label": label,
        }

    def __len__(self):
        return len(self.contrast_drug_label_triples)

    def _validate_contrast_drug_triples_format(self, contrast_drug_label_triples):
        for (contrast, drug_signature_id, label) in contrast_drug_label_triples:
            if not len(drug_signature_id.rsplit("_", 3)) in (3, 4):
                raise ValueError(
                    "Drug contrast name in wrong format: %s" % drug_signature_id
                )


def get_weighted_sampler(labels: List) -> WeightedRandomSampler:
    """
    Implement a weighted sampler that samples (almost) equal numbers of positive and
    negative examples
    :param labels: the list of labels of the training examples, used to calculate the
    sampling probability for each sample
    """
    class_counts = Counter(labels)
    num_samples = len(labels)
    class_weights = [num_samples / class_counts[i] for i in [0, 1]]
    weights = [class_weights[labels[i]] for i in range(int(num_samples))]
    train_sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
    return train_sampler
