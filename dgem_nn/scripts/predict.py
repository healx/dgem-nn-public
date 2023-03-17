import argparse
import logging
import os

from dgem_nn.data_processing.disease_profile import (
    load_omicsoft_diseases_signatures,
    preprocess_diseases_signatures,
)
from dgem_nn.data_processing.drug_profile import load_drugs_signatures_from_dir
from dgem_nn.data_processing.utils import store_predictions_in_tsv, aggregate_predictions_over_models
from dgem_nn.interpretability.explainability_utils import explain
from dgem_nn.networks.bi_encoder import BiEncoder
from dgem_nn.networks.dgem_nn_model import DgemNN
from dgem_nn.networks.dgem_nn_ranker import DgemNNRanker, sum_predictions
from dgem_nn.networks.mlp_baseline import MLPBaseline
from dgem_nn.networks.mlp_cell_line_invariant import MLPCellLineInvariant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict drugs for one or multiple contrasts"
    )
    parser.add_argument("--drugs-signatures-dir")
    parser.add_argument("--diseases-contrasts-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--network-type", default="cell-line-invariant")
    parser.add_argument("--predictions-dir")
    parser.add_argument(
        "--omicsoft-genes-map-tsv", default="data/genes_map_omicsoft_diseases.tsv"
    )
    parser.add_argument(
        "--drug-features-ids-file", default="data/drug_features_ids_l1000.npy"
    )
    parser.add_argument("--l1000-genes-file", default="data/geneinfo_beta.txt")
    parser.add_argument("--ensemble", action="store_true")

    return parser.parse_args()


network_type_map = {
    "model": MLPBaseline,
    "biencoder": BiEncoder,
    "cell-line-invariant": MLPCellLineInvariant,
}
if __name__ == "__main__":

    args = parse_arguments()
    os.makedirs(args.predictions_dir, exist_ok=True)
    disease_signatures = load_omicsoft_diseases_signatures(
        diseases_signatures_path=args.diseases_contrasts_dir,
    )
    disease_signatures = preprocess_diseases_signatures(
        diseases_signatures=disease_signatures
    )
    drugs_signatures = load_drugs_signatures_from_dir(args.drugs_signatures_dir)

    if args.ensemble:
        predictions_per_model_per_contrast = []
        for m_path in os.listdir(args.model_path):
            mlp = DgemNN(
                model_type=network_type_map[args.network_type],
                model_path=os.path.join(args.model_path, m_path)
            )
            dgem_nn_ranker = DgemNNRanker(drugs_signatures=drugs_signatures, model=mlp)

            predictions_per_model_per_contrast.append(
                dgem_nn_ranker.predict(
                    diseases_signatures=disease_signatures,
                    multi_gpu=True,
                    batch_size=8192,
                    topn=50,
                    keep_full_perturbation=False
                )
            )
        predictions_per_contrast = aggregate_predictions_over_models(predictions_per_model_per_contrast)
    else:
        mlp = DgemNN(
            model_type=network_type_map[args.network_type], model_path=args.model_path
        )
        dgem_nn_ranker = DgemNNRanker(drugs_signatures=drugs_signatures, model=mlp)
        predictions_per_contrast = dgem_nn_ranker.predict(
            diseases_signatures=disease_signatures,
            multi_gpu=True,
            batch_size=8192,
            topn=50,
            keep_full_perturbation=False
        )

    for contrast_file in os.listdir(args.diseases_contrasts_dir):
        store_predictions_in_tsv(
            predictions=predictions_per_contrast[contrast_file.rsplit(".", 1)[0]],
            filename=os.path.join(
                args.predictions_dir, contrast_file.rsplit(".", 1)[0] + "_predictions"
            ),
        )
    aggregated_predictions = sum_predictions(
        predictions_per_contrast, sort_ascending=False
    )
    store_predictions_in_tsv(
        predictions=aggregated_predictions,
        filename=os.path.join(
            args.predictions_dir,
            args.diseases_contrasts_dir.rsplit("/", 1)[0] + "_predictions",
        ),
    )
    explain(disease_signatures=disease_signatures,
            drugs_signatures=drugs_signatures,
            predictions=aggregated_predictions,
            omicsoft_genes_map_tsv=args.omicsoft_genes_map_tsv,
            drug_features_ids_file=args.drug_features_ids_file,
            l1000_genes_file=args.l1000_genes_file
            )
