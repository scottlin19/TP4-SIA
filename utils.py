import pandas as pd
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from letters import get_letter_bitmap

def load_csv(file_path,attributes): 
    
    df = pd.read_csv(file_path, names=attributes,skiprows=[0])
    
    features = attributes[1:]
    x = df.loc[:, features].values

    return (df.loc[:,attributes[0]].values,StandardScaler().fit_transform(x)) #scale features 

def store_patterns(patterns): 
    to_return = []
    for pattern in patterns:
        mat = get_letter_bitmap(pattern.upper())
        if mat != None:
            to_return.append([i for row in mat for i in row])
    return to_return


def get_unknown_pattern(stored_patterns, probability, letter, conserve_pattern): 
     
    if letter == -1: 
        i_x = np.random.randint(0, len(stored_patterns))
        unknown_pattern = stored_patterns[i_x].copy()
    else: 
        mat = get_letter_bitmap(letter.upper())
        unknown_pattern = [i for row in mat for i in row]
    count = 0
    for i in range(len(unknown_pattern)): 
        if( probability >= np.random.uniform(0,1)): 
            if conserve_pattern: 
                if(unknown_pattern[i] == -1): 
                    unknown_pattern[i] = 1
                    count += 1
            else: 
                if(unknown_pattern[i] == 1): 
                    unknown_pattern[i] = -1
                else: 
                    unknown_pattern[i] = 1 
                count += 1
                
    print(f"Unknown patterns: {unknown_pattern}")
    print(f"Total modifications: {count}\n")
    return unknown_pattern

def are_orthogonal(stored_patterns, letters_to_store): 
    print("Patterns orthogonality:")
    for i in range(len(stored_patterns)):
        for j in range(i+1,len(stored_patterns)):
            print(f"{letters_to_store[i]} & {letters_to_store[j]}: {np.inner(stored_patterns[i], stored_patterns[j])}") 