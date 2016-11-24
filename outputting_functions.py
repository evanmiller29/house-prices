def output_sub(ID, predictions):
    
    import pandas as pd
    
    pred_df = pd.DataFrame(predictions, index=ID, columns=["SalePrice"])

    print('Outputting predictions..')
    pred_df.to_csv('output.csv', header=True, index_label='Id')