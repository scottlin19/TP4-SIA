import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_csv(file_path,attributes): 
    
    df = pd.read_csv(file_path, names=attributes,skiprows=[0])
    
    features = attributes[1:]
    x = df.loc[:, features].values

    return (df.loc[:,attributes[0]].values,StandardScaler().fit_transform(x)) #scale features 

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

def get_unknown_pattern(stored_patterns, probability, letter): 
     
    if letter == -1: 
        i_x = np.random.randint(0, len(stored_patterns))
    else: 
        i_x = letter_to_index_mapper(letter)
        if i_x == -1: 
            print("Invalid letter")
            exit()
    count = 0
    unknown_pattern = stored_patterns[i_x].copy()
    print(unknown_pattern)
    for i in range(len(unknown_pattern)): 
        if( probability >= np.random.uniform(0,1)): 
            # if(unknown_pattern[i] == 1): 
            #    unknown_pattern[i] = -1
            # else: 
            #     unknown_pattern[i] = 1 
            # count += 1
            if(unknown_pattern[i] == -1): 
               unknown_pattern[i] = 1
               count += 1
    print(unknown_pattern)
    print(f"Modifications: {count}\n")
    return unknown_pattern

def letter_to_index_mapper(letter):
    if letter == "J":
        return 0
    elif letter == "E":
        return 1
    elif letter == "K":
        return 2
    elif letter == "N":
        return 3
    return -1