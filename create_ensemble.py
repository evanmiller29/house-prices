''' 
This function should be sourced in the following way

python create_ensemble.py 2 mean lassocv rf

arg1 = number of models to be included in ensemble. Limited to 4
arg2 = type of voting (only mean/median are included at current)
arg3 = predictions from model 1 to be included
arg4 = predictions from model 2 to be included
arg5 = predictions from model 3 to be included
arg6 = predictions from model 4 to be included    

'''

import sys
import pandas as pd

n_models = int(sys.argv[1])
voting = sys.argv[2]

models = sys.argv[3:3 + n_models]  
test = pd.read_csv('Data/test.csv')

ensemble = pd.DataFrame(index = test['Id'])

ensemble_file = 'Model Outputs/'

for model in models:
    ensemble_file = ensemble_file + model + '_'

ensemble_file = ensemble_file + 'outputs.csv'    
    
for idx, model in enumerate(models):
    print('Reading in predictions from %s' % (model))
    model_file = 'Model Outputs/' + model + '_output.csv'
    model = pd.read_csv(model_file, index_col = 'Id')
    col = 'SP' + str(idx + 1)
    
    ensemble[col] = model['SalePrice']
    
print('Creating an ensemble of the model predictions using %s voting..' % (voting))

if voting == 'mean':
    ensemble['SalePrice'] = ensemble.mean(axis = 1)
    
if voting == 'median':
    ensemble['SalePrice'] = ensemble.median(axis = 1)
    
ensemble = ensemble['SalePrice']

print('Outputting the ensemble..')

ensemble.to_csv(ensemble_file, header=True, index_label='Id')