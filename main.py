from oja import run_oja
from kohonen import run_kohonen
import pandas as pd
import numpy as np
import json, csv
from hopfield import run_hopfield
import utils

with open("config.json") as f:
    config = json.load(f) 
              

exercise = input('Enter the exercise (possible values: 1,2): ')

if exercise == '1':
    
    training_set = utils.load_csv("files/europe.csv",['Country','Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment'])

    type_ = input('Select \'1\' for Kohonen or \'2\' for Oja: ')
    if type_ == '1':
        grid_dimension = config["ej1"]["kohonen"]["grid_dimension"]
        radius_value = config["ej1"]["kohonen"]["radius_value"]
        learning_rate = config["ej1"]["kohonen"]["learning_rate"]
        epochs_amount = config["ej1"]["kohonen"]["epochs_amount"]
        use_input_as_weights = config["ej1"]["kohonen"]["use_input_as_weights"]
        run_kohonen(training_set, grid_dimension, radius_value, learning_rate, epochs_amount,use_input_as_weights)
        
    if type_ == '2':
        learning_rate = config["ej1"]["oja"]["learning_rate"]
        epochs_amount = config["ej1"]["oja"]["epochs_amount"]
        w = run_oja(training_set, learning_rate, epochs_amount)
        print(w)

elif exercise == '2': 

    stored_patterns = utils.store_patterns('files/letters.txt')
    unknown_pattern = utils.get_unknown_pattern(stored_patterns,0.3)

    run_hopfield(stored_patterns,unknown_pattern)
    
else : 
    print("Invalid Input") 
    exit 

