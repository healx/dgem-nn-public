import json
import logging
import os

import numpy
from attrs import define

logger = logging.getLogger(__name__)


@define
class DrugProfile:
    drug_signature: numpy.array
    smiles: str


def build_drugs_names_vs_experiment_map(drugs_signatures: dict) -> dict:
    """
    :return a dict of lists that has as keys the drug names and as values the experiments
    associated with it.
    For example, if we have the following drug signatures: drugA_-5_HL60,drugA_-5_MCF7
    we 'll expect to get {"drugA":["drugA_-5_HL60", "drugA_-5_MCF7"]}
    """
    drug_names_index = {}
    for drug_signature_id in drugs_signatures.keys():
        drug_name = drug_signature_id.split("_")[3]
        if not drug_name in drug_names_index:
            drug_names_index[drug_name] = []
        drug_names_index[drug_name].append(drug_signature_id)
    return drug_names_index


def load_drugs_signatures_from_dir(drugs_signatures_dir, smiles_dict_json="smiles_dict.json"):
    drugs_signatures = {}
    with open(os.path.join(drugs_signatures_dir, smiles_dict_json), "r") as f:
        smiles_dict = json.load(f)
    for f in os.listdir(drugs_signatures_dir):
        if f == smiles_dict_json:
            continue
        signature = numpy.load(os.path.join(drugs_signatures_dir, f))
        pert_id = f.split(".")[0]
        drugs_signatures[pert_id] = DrugProfile(drug_signature=signature,
                                                smiles=smiles_dict[pert_id])
    return drugs_signatures
