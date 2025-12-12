# CS224W Project: ECG Multi-task GNN

This project implements a Multi-task Graph Neural Network (GNN) for detecting Structural Heart Disease (SHD) from ECG data.

## Dataset Setup

**Important:** You need to manually add the dataset to run this project.

The project uses the **EchoNext** dataset. 
You can download it from PhysioNet: [https://physionet.org/content/echonext/1.1.0/](https://physionet.org/content/echonext/1.1.0/)

### Instructions
1.  Download the dataset files.
2.  Create a folder named `Dataset` in the root directory of this project.
3.  Place the following files inside the `Dataset/` directory:
    *   `echonext_metadata_100k.csv`
    *   `EchoNext_train_waveforms.npy`
    *   `EchoNext_val_waveforms.npy`
    *   `EchoNext_train_tabular_features.npy`
    *   `EchoNext_val_tabular_features.npy`
    *   `EchoNext_test_waveforms.npy`
    *   `EchoNext_test_tabular_features.npy`

## Quick Start
See `GraphECG_Multitask_Colab_QuickStart.ipynb` for a walkthrough of the training and evaluation process.
