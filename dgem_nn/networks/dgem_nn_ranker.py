import logging
from typing import List, Union

from dgem_nn.data_processing.omics_dataset import OmicsDataset
from dgem_nn.data_processing.utils import (
    ranked_predictions_per_contrast,
)

logger = logging.getLogger(__name__)


class DgemNNRanker:
    def __init__(
            self,
            drugs_signatures: dict,
            model
    ):
        """
        Wrapper class to predict with a pretrained model on a given disease contrast, over all disease signatures.
        """
        self.model = model
        self.drugs_signatures = drugs_signatures

    def predict(
            self,
            diseases_signatures,
            batch_size=8192,
            multi_gpu=False,
            topn=500,
            verbose=True,
            keep_full_perturbation=False
    ) -> dict:
        """
        Predict a ranking of topn drugs, for a given disease contrast. Results are saved
         in output file.
        """
        drugs_signatures_keys = list(self.drugs_signatures.keys())
        diseases_signatures_keys = list(diseases_signatures.keys())
        if verbose:
            logger.info(
                "Total drugs singatures loaded: %s "
                "First 5 signatures: %s"
                % (len(drugs_signatures_keys), drugs_signatures_keys[:5])
            )

        test_set_triples = [
            [disease_signature, drug_signature_key, 0]
            for drug_signature_key in drugs_signatures_keys
            for disease_signature in diseases_signatures_keys
        ]
        test_dataset = OmicsDataset(
            diseases_signatures=diseases_signatures,
            drugs_signatures=self.drugs_signatures,
            contrast_drug_label_triples=test_set_triples,
        )
        predictions = self.model.predict(
            test_dataset=test_dataset, batch_size=batch_size, multi_gpu=multi_gpu
        )
        ranked_preds_per_contrast = ranked_predictions_per_contrast(
            test_set_triples=test_set_triples,
            predictions=predictions,
            topn=topn,
            keep_full_perturbation=keep_full_perturbation,
        )
        if verbose:
            logger.info(
                "First 5 ranked predictions for 3 random contrasts: %s"
                % [
                      (constrast, preds[:5])
                      for constrast, preds in ranked_preds_per_contrast.items()
                  ][:3]
            )
        return ranked_preds_per_contrast


def rank_predictions(aggregated_predictions, topn, sort_ascending):
    return sorted(
        aggregated_predictions.items(),
        key=lambda item: item[1],
        reverse=not sort_ascending,
    )[:topn]


def sum_predictions(
        predictions_per_contrast: dict,
        sort_ascending: bool,
        topn: int = 500,
        raw: bool = False,
) -> Union[dict, List]:
    aggregated_predictions = {}
    for contrast, predictions in predictions_per_contrast.items():
        for prediction in predictions:
            drug, score = prediction[0], prediction[1]
            if not drug in aggregated_predictions:
                aggregated_predictions[drug] = score
            else:
                aggregated_predictions[drug] += score
    if raw:
        return aggregated_predictions
    else:
        return rank_predictions(aggregated_predictions=aggregated_predictions,
                                topn=topn,
                                sort_ascending=sort_ascending)
