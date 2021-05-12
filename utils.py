import pandas as pd
import json, csv
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_csv(file_path,attributes): 
    
    df = pd.read_csv(file_path, names=attributes,skiprows=[0])
    
    features = attributes[1:]
    x = df.loc[:, features].values

    return StandardScaler().fit_transform(x) #scale features 

def store_patterns(file_path): 
    rows_per_entry = 5

    datafile = open(file_path, 'r')
    datareader = csv.reader(datafile, delimiter=' ')
    data = []
    row_count = 0
    entry = []
    for row in datareader:
        if row_count < rows_per_entry:
            entry += [int(a) for a in row if a != '']
        else:
            row_count = 0
            data.append(entry)
            entry = [int(a) for a in row if a != '']
        row_count += 1

    data.append(entry)

    return data

def get_unknown_pattern(stored_patterns, probability): 
    # 1 --> 0.2 --> -1
    # i_x = np.random.randint(0, len(stored_patterns))
    unknown_pattern = stored_patterns[0].copy()
    print(unknown_pattern)
    for i in range(len(unknown_pattern)): 
        if( probability >= np.random.uniform(0,1)): 
            if(unknown_pattern[i] == 1): 
               unknown_pattern[i] = -1
            else: 
                unknown_pattern[i] = 1 
    print(unknown_pattern)
    return unknown_pattern           