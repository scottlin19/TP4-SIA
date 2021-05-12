from oja import run_oja
from kohonen import run_kohonen
from hopfield import run_hopfield
import utils
import json
with open("config.json") as f:
    config = json.load(f) 
              

exercise = input('Enter the exercise (possible values: 1,2): ')

if exercise == '1':
    file_path = config["ej1"]["file_path"]
    (countries, training_set) = utils.load_csv(file_path,['Country','Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment'])

    type_ = input('Select \'1\' for Kohonen or \'2\' for Oja: ')
    if type_ == '1':
        grid_dimension = config["ej1"]["kohonen"]["grid_dimension"]
        radius_value = config["ej1"]["kohonen"]["radius_value"]
        learning_rate = config["ej1"]["kohonen"]["learning_rate"]
        epochs_amount = config["ej1"]["kohonen"]["epochs_amount"]
        use_input_as_weights = config["ej1"]["kohonen"]["use_input_as_weights"]
        run_kohonen(training_set, grid_dimension, radius_value, learning_rate, epochs_amount,use_input_as_weights,countries)
        
    if type_ == '2':
        learning_rate = config["ej1"]["oja"]["learning_rate"]
        epochs_amount = config["ej1"]["oja"]["epochs_amount"]
        run_oja(training_set, learning_rate, epochs_amount,countries)
        # print(w)

elif exercise == '2': 

    noise_probability = config["ej2"]["hopfield"]["noise_probability"]
    pattern_to_add_noise = config["ej2"]["hopfield"]["pattern_to_add_noise"]
    file_path = config["ej2"]["hopfield"]["file_path"]
    stored_patterns = utils.store_patterns(file_path)
    # expected_index = utils.letter_to_index_mapper(pattern_to_add_noise)
    unknown_pattern = utils.get_unknown_pattern(stored_patterns,noise_probability,pattern_to_add_noise)
    run_hopfield(stored_patterns,unknown_pattern)
    # found, i = run_hopfield(stored_patterns,unknown_pattern)
    # p,fp,n = 0,0,0
    # for i in range(100):
    #     unknown_pattern = utils.get_unknown_pattern(stored_patterns,noise_probability,pattern_to_add_noise)
    #     found, i = run_hopfield(stored_patterns,unknown_pattern)
    #     if found:
    #         if i == expected_index:
    #             p += 1
    #         else:
    #             fp += 1
    #     else:
    #         n += 1
    # print(f"After 100 iterations:\npositives: {p}\t false positives: {fp}\t negatives: {n} ")
else : 
    print("Invalid Input") 
    exit() 

