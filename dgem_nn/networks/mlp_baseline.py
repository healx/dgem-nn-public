import torch
from torch import nn


class MLPBaseline(nn.Module):
    def __init__(
            self,
            compound_signature_input_dim: tuple,
            compound_fingerprint_input_dim: int,
            disease_signature_input_dim: int,
            cell_line_input_dim: int,
            dosage_input_dim: int,
            internal_layer_size: int = 1024,
    ):
        """
        Simple NN architecture:
        We assume that each instance in the dataset contains the following inputs:
        (disease_contrast, drug_signature, drug_fingerprint, cell_line, dosage)
        Each input is encoded as follows:
        - disease signatures are encoded through a convolutional layer since for each
        gene position we have both the log fold change and the pvalue, so the input
        is nr_of_genes x 2.
        - drug signatures are encoded through either a convolutional layer (in case of
        CMAP v1 where we have both the log fold change and the pvalue for each gene, so
        the input is nr_of_genes x 2) or a linear layer (in case of L1000 data where
        only log fold change is given for each gene, so the input is nr_of_genes x 1)
        - drug fingerprints are encoded through a linear layer.

        All the above representations are concatenated and passed through a subsequent
        stack of linear layers.
        """
        super(MLPBaseline, self).__init__()

        self.compound_signature_encoder = nn.Sequential(
            nn.Linear(compound_signature_input_dim[0], internal_layer_size),
            nn.ReLU(),
        )
        self.compound_fingerprints_encoder = nn.Sequential(
            nn.Linear(compound_fingerprint_input_dim, internal_layer_size),
            nn.ReLU(),
        )

        self.disease_encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=disease_signature_input_dim,
                out_channels=internal_layer_size,
                kernel_size=2,
            ),
            nn.ReLU(),
        )

        self.last_linear = nn.Sequential(
            nn.Linear(
                3 * internal_layer_size + cell_line_input_dim + dosage_input_dim, 1
            ),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 1),
        )

    def forward(
            self, disease_contrast, drug_signature, drug_fingerprint, cell_line, dosage
    ):
        x1 = self.disease_encoder(disease_contrast).squeeze(2)
        x2 = self.compound_signature_encoder(drug_signature)
        x3 = self.compound_fingerprints_encoder(drug_fingerprint)
        cat = torch.cat([x1, x2, x3, cell_line, dosage], 1)
        return self.last_linear(cat).squeeze()
