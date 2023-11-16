#utils for data handling

import os
import csv
import yaml
import pandas as pd

def combine_csvs(project_dir, output_file):
    """Combine multiple csv files into one csv file.
        CSVs must have same headers.
    """
    # List all CSV files in the directory
    dataDir = os.path.join(project_dir, "data")
    csv_files = [file for file in os.listdir(dataDir) if file.endswith(".csv")]
    output_path = os.path.join(dataDir, output_file)

    if os.path.exists(output_path):
        print(f"Output file already exists! File will be overwritten.")

    else:
        # Open the output file for writing
        with open(output_path, mode='w', newline='') as combined_csv:
            writer = csv.writer(combined_csv)

            # Write the header from the first CSV
            with open(os.path.join(dataDir, csv_files[0]), 'r') as first_csv:
                reader = csv.reader(first_csv)
                header = next(reader)
                writer.writerow(header)

            # Iterate through all CSV files and append their rows to the combined CSV
            for csv_file in csv_files:
                with open(os.path.join(dataDir, csv_file), 'r') as input_csv:
                    reader = csv.reader(input_csv)
                    next(reader)  # Skip the header
                    for row in reader:
                        writer.writerow(row)

    return output_path, print("Finished combining CSVs!")

def read_data(input_file, index_col = None):
    """Read in a csv file and return a pandas dataframe.
    """
    df = pd.read_csv(input_file, index_col = index_col)
    return df

def create_new_project(project_name, project_dir):
    """Create a new project directory.
    """
    project_path = os.path.join(project_dir, project_name)
    

    if os.path.exists(project_path):
        print("Project directory already exists!")
        return os.path.join(str(project_path), "config.yaml")
    
    os.makedirs(project_path, exist_ok=True)
    data_path = os.path.join(project_path, "data")
    results_path = os.path.join(project_path, "results")
    model_path = os.path.join(project_path, "models")

    for p in [data_path, results_path, model_path]:
        os.makedirs(p, exist_ok=True)

    # Create config file
    config_path = os.path.join(project_path)
    create_config(project_name, config_path)

    return project_path, print("Finished creating new project!")

def save_to_yaml(data, filename):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
    return filename

def create_config(project_name, project_dir):
    """
    Create a config file with prefilled values
    """

    #create outline for config file
    project_info = {
        'project_name': project_name,
        'project_path': project_dir,
    }
    glm_params = {
        
        'predictors': [
            'predictor1',
            'predictor2',
            'predictor3'
        ],
        'predictors_shift_bounds_default': [-50, 100],
        'predictors_shift_bounds': {
            'predictor1': [-2, 2],
            'predictor2': [-2, 2],
            'predictor3': [-2, 2],
        },
        'response': 'photometryNI',
        'type': 'Normal',
        'glm_keyword_args': {
            'alpha': 0.5,
            'l1_ratio': 0.5, #or list if using elasticnetcv
            'fit_intercept': True,
            'max_iter': 1000,            
            'warm_start': False,
            'selection': 'cyclic', #or random
            'score_metric': 'r2', #options: 'r2', 'mse', 'avg'
            'cv': 5, #number of cross validation folds
            'n_alphas': 100, #number of alphas to test
            'n_jobs': -1, #number of jobs to run in parallel
            },
        'ridge_keyword_args': {
            'alpha': 0.5,
            'fit_intercept': True,
            'max_iter': 1000,            
            'solver': 'auto', #options: 'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'
            'gcv_mode': 'auto', #options: None, 'auto', 'svd', 'eigen'
            'score_metric': 'r2', #options: 'r2', 'mse', 'avg'
            'cv': 5, #number of cross validation folds
            'n_jobs': -1, #number of jobs to run in parallel
            },
    }
    train_test_split = {
        'train_size': 0.8,
        'test_size': 0.2,
    }

    data = {'Project': project_info,
            'glm_params': glm_params,
            'train_test_split': train_test_split,}

    cfg_file = os.path.join(project_dir, "config.yaml")
    save_to_yaml(data, cfg_file)
    return cfg_file

def load_config(config_file):
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config