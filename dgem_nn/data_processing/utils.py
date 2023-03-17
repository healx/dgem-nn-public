import logging
import statistics
from functools import reduce
from typing import List

import deepchem
import pandas
import torch

logger = logging.getLogger(__name__)


def ranked_predictions_per_contrast(
        test_set_triples: List,
        predictions: List,
        topn: int,
        keep_full_perturbation: bool = False,
) -> dict:
    """
    Rank the predicted drugs by aligning and sorting contrasts, drug perturbation and predictions.
    :param test_set_triples: triples from the test set of the form (contrast, drug_perturbation, label)
    :param predictions: list of predictions, softmax outputs
    :topn: top predictions to keep per contrast
    :param keep_full_perturbation: bool to configure if we will keep the full
    perturbation name (which includes dosage, cell line and drug) or just the drug name.
    """

    ranked_preds_per_contrast = {}
    for (contrast, drug_perturbation, _), prediction in zip(
            test_set_triples, predictions
    ):
        if not contrast in ranked_preds_per_contrast:
            ranked_preds_per_contrast[contrast] = {}
        if keep_full_perturbation:
            ranked_preds_per_contrast[contrast][drug_perturbation] = prediction
        else:
            drug_fields = drug_perturbation.split("_")
            drug = drug_fields[0] + "_" + drug_fields[3]
            if (
                    not drug in ranked_preds_per_contrast[contrast]
            ) or (  # keep the highest prediction
                    ranked_preds_per_contrast[contrast][drug] < prediction
            ):
                ranked_preds_per_contrast[contrast][drug] = prediction

    for contrast in ranked_preds_per_contrast.keys():
        ranked_preds_per_contrast[contrast] = sorted(
            ranked_preds_per_contrast[contrast].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:topn]
    return ranked_preds_per_contrast


def save_predictions(predictions: List, test_file: str, out_file: str):
    """Save predictions as a tsv by adding an extra column in the test file"""

    df = pandas.read_csv(test_file, sep="\t")
    df["predictions"] = predictions
    df.to_csv(out_file, sep="\t")


def encode_smiles_into_fingerprints(
        smiles: list, fingeprint_vector_size: int = 2048
) -> dict:
    """
    Encode a list of SMILES strings as Morgan fingerprints
    :param smiles: the list of SMILES strings
    :param fingeprint_vector_size: size of the Morgan representation.
    :return dictionary of representations
    """
    compounds_encoder = deepchem.feat.CircularFingerprint(size=fingeprint_vector_size)
    tensorized_fingerprints = torch.from_numpy(
        compounds_encoder.featurize(smiles)
    ).float()
    return {
        smile: tensorized_fingerprint
        for smile, tensorized_fingerprint in zip(smiles, tensorized_fingerprints)
    }


def load_as_tensors(vectors: dict):
    """ """
    tensors = {}
    for key, vector in vectors.items():
        if type(vector) is list:
            tensors[key] = torch.Tensor(vector)
        else:
            tensors[key] = torch.from_numpy(vector).float()
    return tensors


def group_across_dictionaries(dictionaries_list):
    common_keys = list(reduce(lambda x, y: x & y.keys(), dictionaries_list))
    avg_dict = {k: statistics.mean([d[k] for d in dictionaries_list]) for k in common_keys}
    return avg_dict


def build_triples_for_prediction(diseases_signatures_keys, drugs_signatures_keys):
    triples = []
    for disease_signature_key in diseases_signatures_keys:
        for drug_signature_key in drugs_signatures_keys:
            triples.append([disease_signature_key, drug_signature_key, 0])
    return list(set(triples))


def store_predictions_in_tsv(predictions, filename):
    with open(filename, "w") as f:
        f.write("drug\tscore\n")
        for (drug, score) in predictions:
            f.write("%s\t%s\n" % (drug, score))


def aggregate_predictions_over_models(predictions_per_contrast: list, aggregate_different_brd_ids_by_drug_name=False):
    aggregated_predictions_per_contrast = {}
    for contrast_predictions in predictions_per_contrast:
        for contrast, predictions in contrast_predictions.items():
            if not contrast in aggregated_predictions_per_contrast:
                aggregated_predictions_per_contrast[contrast] = {}
            for (drug_brd, score) in predictions:
                if aggregate_different_brd_ids_by_drug_name:
                    drug = drug_brd.split("_")[0]
                else:
                    drug = drug_brd
                if not drug in aggregated_predictions_per_contrast[contrast]:
                    aggregated_predictions_per_contrast[contrast][drug] = score
                else:
                    aggregated_predictions_per_contrast[contrast][drug] += score

    return {contrast: list(d.items()) for contrast, d in aggregated_predictions_per_contrast.items()}
