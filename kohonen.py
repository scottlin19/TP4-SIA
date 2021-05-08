import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# initialize k, eta, w, r
# select row 
# search winning neuron 
# update neighbours weights, eta and r

class KohonenNeuron:
    def __init__(self,weights):
        self.weights = weights

    def __repr__(self):
        return f"Neuron - w: {self.weights}\n"
    
    def __str__(self):
        return f"Neuron - w: {self.weights}\n"


class Kohonen:
    def __init__(self, training_set, grid_dimension, radius, input_weights, learning_rate=1):
        self.training_set = training_set
        self.input_dimension = len(training_set[0])
        self.grid_dimension = grid_dimension
        self.learning_rate = learning_rate

        # if(radius[0] == true):
        #     self.radius = radius[1]
        # else: 
        #     self.radius = grid_dimension
        # self.radius = radius    
        self.radius = grid_dimension
        self.neurons = [None] * grid_dimension
        #initialize weights 
        for i in range(self.grid_dimension):
            self.neurons[i] = [None] * grid_dimension
            for j in range(self.grid_dimension):
                if input_weights == False:
                    w = np.random.default_rng().uniform(0,1,self.input_dimension) #TODO cambiar a la otra opcion con un flag
                else: 
                    i_x = np.random.randint(0, len(self.training_set))               # get random input
                    input_ = self.training_set[i_x]
                    w = input_.copy()
                self.neurons[i][j] = KohonenNeuron(w)

    def train(self,epochs):
        tr_length = len(self.training_set)
        activations = np.zeros((self.grid_dimension, self.grid_dimension))

        for i in range(epochs):
            aux_training = self.training_set.copy()
            self.learning_rate = 1                                          # restart el eta
            aux_radius = self.radius
            while len(aux_training) > 0: 
                i_x = np.random.randint(0, len(aux_training))               # get random input
                input_ = aux_training[i_x]
                aux_training = np.delete(aux_training, i_x, axis=0)

                
                (x,y,winner_neuron) = self.get_winner_neuron(input_)        # search winner neuron 
                activations[x][y] += 1
                self.update_weights(x,y,winner_neuron,input_)               # update neighbours and curr neuron weights 
                
                if(aux_radius > 1):
                    aux_radius -= 1
                self.learning_rate = 1/(tr_length - len(aux_training))
         

        # Graficar matriz U  
        print("---------------- END ------------------")

        # Graficar mapa de calor con activaciones
        print(activations)
        ax = sns.heatmap(activations, linewidth=0.5)
        # plt.imshow(activations, cmap='hot', interpolation='nearest')
        # plt.show()
     
    def update_weights(self,x,y,winner_neuron,input_):
        # Iterar por todos las neuronas
            # Calcular su distancia al ganador 
            # Si dist < R: cambiar peso (es vecina)
            # Sino nada
        for i,row in enumerate(self.neurons):
            for j,neuron in enumerate(row): 
                if self.input_distance((i,j), (x,y)) <= self.radius: #x,y es la pos de la winner neuron -->  es vecina
                    neuron.weights += self.learning_rate * (input_ - neuron.weights)
                    print(f"New wight for neuron({i},{j}): {neuron.weights}")

    def get_winner_neuron(self,input_):
        to_return = None # neuron position in output grid 
        min_dist = float('inf')
        for i,row in enumerate(self.neurons):
            for j,neuron in enumerate(row):
                
                dist = self.input_distance(input_, neuron.weights)
                if dist < min_dist: 
                    min_dist = dist
                    winner_neuron = self.neurons[i][j]
                    to_return = (i,j,winner_neuron)

        print(f"Returning - x: {to_return[0]} - y: {to_return[1]}")
        return to_return

            
    def input_distance(self,p1,p2):
        dif_squares = [(v1 - v2)**2 for v1,v2 in zip(p1,p2)]
        return math.sqrt(sum(dif_squares))
    
def run_kohonen(training_set, grid_dimension, radius,learning_rate,epochs, use_input_as_weights):
    kohonen = Kohonen(training_set, grid_dimension, radius,use_input_as_weights,learning_rate)
    kohonen.train(epochs)