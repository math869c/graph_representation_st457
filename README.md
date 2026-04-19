# graph_representation_st457

Link to group project description: https://docs.google.com/document/d/1jb6vDvd4IwL1pwYKUXJppwRfqknrucCh9wloeb-G9kg/edit?usp=sharing

Structure of the folder:
data_folder:
* Adjacency_matric.npy: contains adjacency matric
* firm_industry.json: contains graph information about companies
* open_prices_interp.csv: contains all stock prices

Py-files:
* data_maker.py: Creates all files in data_folder 
* helper_functions.py: Contains fucntions for final data processing, training functions for all models, metric functinos and plotting functions
* model_classes.py: Contains the classes for LSTM, TGC, and GAT

ipynb-files, can be split up into three parts:
Part 1, model_ipynb, in folder model_ipynb:
* LSTM_model.ipynb: This runs a baseline of each model to ensure they work and where one can test with simple changes 
* TGC_model.ipynb: This runs a baseline of each model to ensure they work and where one can test with simple changes 
* GAT_model.ipynb: This runs a baseline of each model to ensure they work and where one can test with simple changes 

Part 2, Tune_models, in folder Tune_models_ipynb:
* LSTM_tune.ipynb: Tune the model and find the optimal hyper-parameters 
* TGC_tune.ipynb: Tune the model and find the optimal hyper-parameters 
* GAT_tune.ipynb: Tune the model and find the optimal hyper-parameters 

Part 3, compare models:
* master.ipynb: This combines the best parameters of all models