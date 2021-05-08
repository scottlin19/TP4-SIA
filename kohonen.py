import math
import random
import numpy as np

# initialize k, eta, w, r
# select row 
# search winning neuron 
# update neighbours weights, eta and r

class KohonenNeuron:
    def __init__(self,weights):
        self.weights = weights


class Kohonen:
    def __init__(self, training_set, grid_dimension, radius, learning_rate=1):
        self.training_set = training_set
        self.input_dimension = len(training_set[0])
        self.grid_dimension = grid_dimension
        self.learning_rate = learning_rate

        if(radius[0] == true):
            self.radius = radius[1]
        else: 
            self.radius = grid_dimension
            
        self.neurons = []
        #initialize weights 
        for i in range(self.grid_dimension):
            self.neurons[i] = []
            for j in range(self.grid_dimension):
                w = np.random.default_rng().uniform(0,1,self.input_dimension) #TODO cambiar a la otra opcion con un flag
                self.neurons[i][j] = KohonenNeuron(w)

    def train(self,epochs):
        print(self.neurons)
        
        for i in range(epochs):
            aux_training = self.training_set.copy()

            while len(aux_training) > 0: 
                i_x = np.random.randint(0, len(aux_training))               # get random input
                input_ = aux_training[i_x]
                aux_training = np.delete(aux_training, i_x, axis=0)

                
                (x,y,winner_neuron) = self.get_winner_neuron(input_) # search winner neuron 

                update_weights(x,y,winner_neuron)               # update neighbours and curr neuron weights 
          
    #[raiz(2)  1    raiz(2)      
    # 1        0    1         2
    #]raiz(2)  1    raiz(2)   x
    # raiz((abs(x-x'))^2 + (abs(y-y'))^2)

    # (3,2)    (1,2,3,4,5,6,7)
    # (2,1)
    #raiz((3-2)^2 + (2,1)^2)
    # (4,4)
    #raiz((1)^2 + (2)^2)

     
    def update_weights(x,y,winner_neuron):
        # Iterar por todos las neuronas
            # Calcular su distancia al ganador 
            # Si dist < R: cambiar peso (es vecina)
            # Sino nada
    def get_neighbours(winner_neuron):
         


    def get_winner_neuron(self,input_):
        to_return = None # neuron position in output grid 
        for i,row in enumerate(self.neurons):
            for j,neuron in enumerate(row):
                
                dist = input_distance(input_, neuron.weights)
                if dist < min_dist: 
                    min_dist = dist
                    winner_neuron = neurons[i][j]
                    to_return = (i,j,winner_neuron)
        return to_return

            
    def input_distance(p1,p2):
        dif_squares = [(v1 - v2)**2 for v1,v2 in zip(p1,p2)]
        return math.sqrt(sum(dif_squares))
    
def run_kohonen(training_set, grid_dimension, radius,learning_rate,epochs):
    kohonen = Kohonen(training_set, grid_dimension, radius,learning_rate)
    kohonen.train(epochs)
