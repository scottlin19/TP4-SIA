from oja import run_oja
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler


with open("config.json") as f:
    config = json.load(f)

learning_rate = config["learning_rate"]
epochs_amount = config["epochs"]

def load_csv(): 
    #return np.loadtxt(open("files/europe.csv", "rb"), delimiter=",", skiprows=1, usecols=np.arange(1,7))

    df = pd.read_csv("files/europe.csv", names=['Country','Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment'],skiprows=[0])
    
    features = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
    x = df.loc[:, features].values

    return StandardScaler().fit_transform(x) #scale features 

exercise = input('Enter the exercise (possible values: 1,2): ')

if exercise == '1':
    
    training_set = load_csv()

    type_ = input('Select \'1\' for Kohonen or \'2\' for Oja: ')
    if type_ == '2':
        w = run_oja(training_set, learning_rate, epochs_amount)
        print(w)


