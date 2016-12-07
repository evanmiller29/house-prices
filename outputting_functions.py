def output_sub(ID, predictions, model, submission):
    
    import pandas as pd
    
    pred_df = pd.DataFrame(predictions, index=ID, columns=["SalePrice"])
    
    if submission == 'submission':
        output_suffix = 'output.csv'
    
    if submission == 'stacking':
        output_suffix = 'logged.csv'
    
    print('Outputting predictions..')
    output_file = 'Model outputs/' + model + '_' + output_suffix
    pred_df.to_csv(output_file, header=True, index_label='Id')