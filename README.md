# DGEM-NN

Disease-drug transcriptomics-based matching with deep learning

For more information see our preprint:

### Running example

After cloning and pip installing requirements.txt:

- download the trained models and data
  from [here](https://drive.google.com/drive/folders/17AudOv5Q1JarkIimEg6600_o_WUHkQTY?usp=sharing)
- unzip and place them in the data folder
- run the code as follows:
  ```PYTHONPATH=. python3 dgem_nn/scripts/predict.py --drugs-signatures-dir data/drugs_signatures --diseases-contrasts-dir data/sample_contrasts --model-path trained_models --predictions-dir ps --ensemble```

### Cite

arxiv - url
