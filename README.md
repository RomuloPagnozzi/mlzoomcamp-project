# mlzoomcamp-project
Data source: https://github.com/feedzai/bank-account-fraud

The business problem is predicting bank account fraud, with the target being the 'fraud_bool' column in the dataset.
The model used only the base version of the dataset.(See data source for more info)

The notebook uses kaggle API to download the dataset, ensure to have valid credentials to download it.

The latest version of the model is saved as 'xgb_13aucpr.pkl'
There's a docker image that was uploaded to dockerhub and can be downloaded using:

> docker pull romulopagnozzi/mlzoomcamp-project
