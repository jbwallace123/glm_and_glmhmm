sglm
==============================

A GLM Pipeline for Neuroscience Analyses

Built using sklearn.ElasticNet,Ridge, ElasticNetCV, and RidgeCV

All necessary packages listed in requirements.txt and are pip installable! 

To get started, please take a look at the fitGLM jupyter notebook. When you have 
completed running the notebook the output will generate a project directory.

```bash
Project directory will be created to include: 
|
| Project_Name
| ├── data
|   ├── 00001.csv
|   ├── 00002.csv
|   └── combined_output.csv
| ├── models
|   └── project_name.pkl
| ├── results
|   ├── model_fit.png
|   ├── predicted_vs_actual.png
|   └── residuals.png
| config.yaml
```

data folder: will include all of your data in .csv format. Please refer to the notebook for formatting.

models folder: will include outputs saved from the model_dict.

results folder: includes some figures for quick visualization. 

config.yaml: your config file to set your parameters for the model. 
