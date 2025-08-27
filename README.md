<h1 align="center"><b>SSPNet</b></h1>

Official implementation for our paper:

*SSPNet: Leveraging Robust Medication Recommendation with History and Knowledge*

Haodi Zhang, Jiawei Wen, Jiahong Li, Yuanfeng Song, Liang-Jie Zhang*, Lin Ma* (* denotes correspondence)

*IJCAI 2025 : International Joint Conference on Artificial Intelligence*


## Folder Specification

- `data/` folder contains necessary data or scripts for generating data.
  - `drug-atc.csv`, `ndc2atc_level4.csv`, `ndc2rxnorm_mapping.txt`: mapping files for drug code transformation
  - `atc2rxnorm.pkl`: It maps ATC-4 code to rxnorm code and then query to drugbank.
  - `drug-DDI.csv`: A file containing the drug DDI information which is coded by CID. This file is large and you can download it from [https://drive.google.com/file/d/1s3sHmz9ueVA8YAGTARY8jwrhRdRvVaXs/view?usp=sharing](https://drive.google.com/file/d/1s3sHmz9ueVA8YAGTARY8jwrhRdRvVaXs/view?usp=sharing).
  - `ddi_mask_H.pkl`:  A mask matrix containing the relations between molecule and substructures. If drug molecule $i$ contains substructure $j$, the $j$-th column of $i$-the row of the matrix is set to 1.
  - `ddi_mask_H.py`: The python script responsible for generating `ddi_mask_H.pkl` and `substructure_smiles.pkl`.
  - `processing.py`: The python script responsible for generating `voc_final.pkl`, `records_final.pkl`, `data_final.pkl` and `ddi_A_final.pkl`.    
- `src/` folder contains all the source code.
  - `modules/`: Code for model definition.
  - `utils.py`: Code for metric calculations and some data preparation.
  - `training.py`: Code for the functions used in training and evaluation.
  - `main.py`: Train or evaluate our MoleRec Model.

## Datasets
Under MIMIC Dataset policy, we are not allowed to distribute the datasets. Practioners could go to https://physionet.org/content/mimiciii/1.4/ and requrest the access to [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) and [MIMIC-IV](https://physionet.org/content/mimiciv/3.0/) dataset and then run our processing script to get the complete preprocessed dataset file.


## Acknowledgement
We sincerely thank these repositories [MoleRec](https://github.com/yangnianzu0515/MoleRec) and [SafeDrug](https://github.com/ycq091044/SafeDrug) for their well-implemented pipeline upon which we build our codebase.
