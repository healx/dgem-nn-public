import logging
import os
import time
from statistics import mean

import gseapy
import numpy
import pandas

from dgem_nn.data_processing.drug_profile import build_drugs_names_vs_experiment_map
from dgem_nn.data_processing.utils import group_across_dictionaries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("gseapy").setLevel(logging.WARNING)


def find_significant_features(features_map, contrast, topn=500):
    significant_features = \
        {
            features_map[i].upper(): log_fold_change
            for i, log_fold_change in enumerate(contrast)
            if abs(log_fold_change) >= 0.5
        }
    return dict(sorted(significant_features.items(), key=lambda item: abs(item[1]), reverse=True)[:topn])


def aggreggate_disease_significant_features_across_contrasts(diseases_signatures,
                                                             omicsoft_genes_map_tsv,
                                                             topn=500):
    disease_features_map = (
        pandas.read_csv(omicsoft_genes_map_tsv, sep="\t", dtype={"GeneIndex": int})
        .set_index("GeneIndex")["Name"]
        .to_dict()
    )
    disease_significant_features = {}
    for contrast_id, signature in diseases_signatures.items():
        contrast_significant_features = \
            find_significant_features(
                features_map=disease_features_map,
                contrast=signature,
                topn=topn
            )
        for gene, score in contrast_significant_features.items():
            if not gene in disease_significant_features:
                disease_significant_features[gene] = []
            disease_significant_features[gene].append(score)

    return dict(sorted(disease_significant_features.items(), key=lambda item: mean(item[1]), reverse=True))


def explain(disease_signatures,
            drugs_signatures,
            predictions,
            omicsoft_genes_map_tsv,
            l1000_genes_file,
            drug_features_ids_file):
    disease_significant_features = \
        aggreggate_disease_significant_features_across_contrasts(
            diseases_signatures=disease_signatures,
            omicsoft_genes_map_tsv=omicsoft_genes_map_tsv
        )
    drug_features_ids = numpy.load(drug_features_ids_file)
    drug_features_map = pandas.read_csv(l1000_genes_file, sep="\t",
                                        usecols=["gene_id",
                                                 "gene_symbol"]).set_index(
        "gene_id")["gene_symbol"].to_dict()
    drug_features_map = [drug_features_map[i] for i in drug_features_ids]
    drugs_perturbations_map = build_drugs_names_vs_experiment_map(
        drugs_signatures)

    for (drug_id, score) in predictions:
        perturbations_significant_features = []
        for drug_signature_id in drugs_perturbations_map[drug_id.split("_")[1]]:
            drug_signature = drugs_signatures[drug_signature_id].drug_signature
            perturbations_significant_features.append(
                find_significant_features(
                    features_map=drug_features_map, contrast=drug_signature,
                    topn=500
                )
            )
        drug_significant_features = group_across_dictionaries(
            perturbations_significant_features)
        if not drug_significant_features:
            print(drug_id, "No common genes!")
            continue

        common_genes_keys = list(
            disease_significant_features.keys() & drug_significant_features.keys())
        common_genes = {
            gene: (drug_significant_features[gene], disease_significant_features[gene])
            for gene in common_genes_keys}
        print("=======================")
        logger.info("Common genes with %s: %s" % (drug_id, common_genes))
        if drug_significant_features and disease_significant_features:
            # print(drug_significant_features)
            try:
                common_pathways = find_common_pathways(
                    drug_significant_features=list(drug_significant_features.keys()),
                    disease_significant_features=list(disease_significant_features.keys()),
                    drug_name=drug_id,
                    disease_name="disease")
                logger.info("Common pathways: %s" % common_pathways)
            except Exception as e:
                print(e)
        print("=======================")


def find_common_pathways(drug_significant_features, disease_significant_features, drug_name, disease_name):
    os.makedirs("results", exist_ok=True)
    drug_pathways = gseapy.enrichr(gene_list=drug_significant_features, gene_sets=[
        'GO_Biological_Process_2021',
        # 'GO_Molecular_Function_2021',
        # 'GO_Cellular_Component_2021',
        # 'Reactome_2022',
        # 'WikiPathway_2021_Human'
    ], outdir=os.path.join("results", drug_name), cutoff=0.3).results
    drug_pathways = drug_pathways[drug_pathways["Adjusted P-value"] < 0.3][["Term", "Genes"]].drop_duplicates()
    time.sleep(0.1)
    disease_pathways = gseapy.enrichr(gene_list=disease_significant_features, gene_sets=[
        'GO_Biological_Process_2021',
        # 'GO_Molecular_Function_2021',
        # 'GO_Cellular_Component_2021',
        # 'Reactome_2022',
        # 'WikiPathway_2021_Human'
    ], outdir=os.path.join("results", disease_name), cutoff=0.3).results
    disease_pathways = disease_pathways[disease_pathways["Adjusted P-value"] < 0.3][["Term", "Genes"]].drop_duplicates()
    time.sleep(0.1)
    return pandas.merge(drug_pathways, disease_pathways, on="Term")
