'''Loads trained models and prints performance accuracy on training and validation data'''

import h2o
import time

now = time.strftime("%c")

# Initialize h2o cluster
h2o.init(nthreads=-1,
         max_mem_size='6G')
h2o.remove_all()

# Load Models
path = '/Users/LiamRoberts/rossmann_retail/models/'

# Random Forest
forest_path = 'DRF_model_python_1543868168054_1'
rdf_model = h2o.load_model(f'{path}{forest_path}')

# XG Boost
xg_path = 'XGBoost_model_python_1543866507146_1'
xg_model = h2o.load_model(f'{path}{xg_path}')

# Deep Learning Model
dl_path = 'DeepLearning_model_python_1543866712628_1'
dl_model = h2o.load_model(f'{path}{dl_path}')

# Assign File Path for Saving Data
filepath = 'model_results.txt'

# Create Function to Save Model Reults and Time of Test
def save_results(models,names,filepath):

    with open(filepath,'a') as f:
        f.write("\n")
        f.write(f'{time.strftime("%c")}')
        f.write("\n")
        for model,name in zip(models,names):
                f.write("\n")
                f.write(f'{name}')
                f.write("\n")
                f.write(f'Train RMSE: {model.rmse(train=True)}')
                f.write("\n")
                f.write(f'Validation RMSE: {model.rmse(valid=True)}')
                f.write("\n")
    print('----File Successfully Saved----')
    with open(filepath,'r') as f:
        print(f.read())
    return

# Varbiables to Pass into Function
models = [rdf_model,
          xg_model,
          dl_model]

names = ['Random Forest Model',
         'XGBoost Model',
         'Deep Learning Model']

# Save Results
save_results(models,
             names,
             filepath)

# Shutdown Cluster
h2o.cluster().shutdown()