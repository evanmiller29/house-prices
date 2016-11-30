def output_sub(ID, predictions, model):
    
    import pandas as pd
    
    pred_df = pd.DataFrame(predictions, index=ID, columns=["SalePrice"])

    print('Outputting predictions..')
    output_file = 'Model outputs/' + model + '_' + 'output.csv'
    pred_df.to_csv(output_file, header=True, index_label='Id')