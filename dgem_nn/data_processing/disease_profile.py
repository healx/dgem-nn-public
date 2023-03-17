import logging
import os

import numpy
import pandas

logger = logging.getLogger(__name__)

MAX_NR_OF_GENES_IN_DISEASE_CONTRAST = 30374


def load_omicsoft_disease_profile(
        disease_file: str, max_nr_of_genes_in_disease_contrast: int
) -> numpy.array:
    """
    Load an omicsoft disease contrast
    :param disease_file: we assume that this is an omicsoft disease contrast, that has
    been preprocessed from the ComparisonData file.
    Returns a numpy array of Nx2 dimensions where N is the maximum number of measured
    genes and each position n, n \in N, has the adjusted pvalue in n[0] and the logfold
    change in n[1].
    """
    disease_signature_df = pandas.read_csv(
        disease_file,
        dtype={"Log2FoldChange": float, "AdjustedPValue": float, "GeneIndex": int},
        usecols=["GeneIndex", "AdjustedPValue", "Log2FoldChange"],
    ).drop_duplicates()
    disease_signature_df = disease_signature_df.groupby(
        "GeneIndex", as_index=False
    ).mean()
    disease_signature_df = disease_signature_df.set_index("GeneIndex").reindex(
        range(max_nr_of_genes_in_disease_contrast)
    )
    disease_signature_df["Log2FoldChange"].fillna(value=0.0, inplace=True)
    disease_signature_df["AdjustedPValue"].fillna(value=1.0, inplace=True)

    return disease_signature_df[["AdjustedPValue", "Log2FoldChange"]].to_numpy()


def load_omicsoft_diseases_signatures(
        diseases_signatures_path: str,
        max_nr_of_genes_in_disease_contrast: int = MAX_NR_OF_GENES_IN_DISEASE_CONTRAST,
) -> dict:
    """
    Load all disease signatures from disease signatures folder. This folder has been
    constructed previously by combining omicsoft data, specifically the ComparisonData and
    Comparisons files.
    """

    diseases_signatures = {}
    logger.info("Reading diseases signatures")
    for f in os.listdir(diseases_signatures_path):
        disease_contrast_id = f.rsplit(".", 1)[0]
        disease_signature = load_omicsoft_disease_profile(
            disease_file=os.path.join(diseases_signatures_path, f),
            max_nr_of_genes_in_disease_contrast=max_nr_of_genes_in_disease_contrast,
        )
        diseases_signatures[disease_contrast_id] = disease_signature
    logger.info("loaded %s signatures" % len(diseases_signatures))
    return diseases_signatures


def preprocess_diseases_signatures(diseases_signatures):
    disease_signatures_with_significant_fold_changes = {}
    for disease_signature, expr in diseases_signatures.items():
        disease_signatures_with_significant_fold_changes[disease_signature] = [
            round(v[1], 4) if (v[0] < 0.3) else 0 for v in expr
        ]
    return disease_signatures_with_significant_fold_changes
