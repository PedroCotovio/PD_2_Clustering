import pandas as pd
import os

def get_path(path='dataset'):
    
    pth = os.getcwd()
    pth = os.path.join(pth,path)
    print('Path to Data: {}'.format(pth))
    return pth

def validate_format(df, rows=110, columns=5147, target='DIAGNOSIS', drop=['ID'], col_print=5):
    
    df = pd.DataFrame(df)
    rows = int(rows)
    columns = int(columns)
    
    extra = ''
    
    if columns > col_print:
        extra = 'First {} '.format(col_print)
        
    
    try:
        y = df[target]
        drop.append(target)
        df.drop(drop, axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        if df.shape == (rows, columns):
        
            X = df.values
            
            print('Valid Format')
            print('')
            print('{}Columns: {}'.format(extra, list(df.columns[:col_print])))
            print('')
            print('Targets: {}'.format(set(y)))
            
            df['target'] = y
            
            return df, X, y
    
        else:
            raise AttributeError('No Valid Format Found')
            
    except KeyError:
        print("Dataframe already processed or columns don't exist")