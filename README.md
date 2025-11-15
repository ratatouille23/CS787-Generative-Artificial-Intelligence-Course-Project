# CS787-Generative-Artificial-Intelligence-Course-Project
Improving Hybrid Attention Network (HAN) for Stock Movement Prediction using FinBERT-enhanced Embeddings

# Reproducibility & Artifacts
### Packages and Versions

*   python == 3.12
*   torch == 2.6.0+cu124
*   torcheval == 0.0.7
*   transformers == 4.57.1
*   seaborn == 0.13.2
*   numpy == 2.3.4
*   gensim == 4.4.0
*   matplotlib == 3.10.7
*   pandas == 2.3.3
*   scikit-learn == 1.7.2
*   cuda == 12.4
    
### Steps to Run 

1. Clone the github repository.
2. In the command line, setup a python virtual environment.
3. Install the packages mentioned above.
4. Edit the config files of both the models depending on the system.
5. To run original HAN
   5.1 Type the command "python original_han_dataset.py" to load the datset.
   5.2 Type the command "python train_original_han.py" to train and test the model.
6. To run the updated HAN
   6.1 Type the command "python dataset_finbert.py" to load the datset.
   6.2 Type the command "python train.py" to train and test the model.
