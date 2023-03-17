from setuptools import find_packages, setup

config = {
    "version": "0.0.1",
    "name": "dgem-nn",
    "description": "NN for disease-drug transcriptomics matching",
    "url": "https://www.healx.io",
    "packages": find_packages(
        include=[
            "dgem_nn",
            "dgem_nn.*",
        ]
    ),
    "install_requires": [
        "attrs==22.1.0",
        "pandas==1.4.3",
        "torch==1.12",
        "deepchem==2.6.1",
        "scikit-learn",
        "tqdm",
        "gseapy==1.0.4",
    ],
}

setup(**config)
