import torch
from torch import nn


class MLPCellLineInvariant(nn.Module):
    def __init__(
            self,
            compound_signature_input_dim: tuple,
            disease_signature_input_dim: int,
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
        super(MLPCellLineInvariant, self).__init__()

        self.compound_signature_encoder = nn.Sequential(
            nn.Linear(compound_signature_input_dim[0], internal_layer_size),
            nn.ReLU(),
        )

        self.disease_encoder = nn.Sequential(
            nn.Linear(disease_signature_input_dim, internal_layer_size),
            nn.ReLU(),
        )

        self.last_linear = nn.Sequential(
            nn.Linear(2 * internal_layer_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, disease_contrast, drug_signature):
        x1 = self.disease_encoder(disease_contrast)
        x2 = self.compound_signature_encoder(drug_signature)
        cat = torch.cat([x1, x2], 1)
        return self.last_linear(cat).squeeze()
