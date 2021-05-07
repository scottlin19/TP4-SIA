import math
import random
import numpy as np

class OjaPerceptron:

    def __init__(self, training_set, learning_rate):
        self.training_set = np.array(training_set) 
        self.learning_rate = learning_rate
    
    def train(self, epochs_amount):
        registers_amount = len(self.training_set.T[0])
        dimension = len(self.training_set[0])  # ( (area1,b,c,d,e,f,g) , (area2,2,3,4,5,6,7) ) 
        w = np.random.uniform(0, 1, registers_amount) # array de longitud p+1 con valores random entre 0 y 1  
    
        for epochs in range(epochs_amount):
                
            for input in self.training_set.T: # iterate by columns
                # print(input)
                y = np.inner(input, w) # inner product: sum (input[i]*w_i) 
                
                delta_w = self.learning_rate * y * (input - y*w) # eta* (y*x - y^2 * w ) = eta * y(x - yw)

                w += delta_w 
    

        norm = math.sqrt(np.inner(w,w))

        return w / norm  # PC1


def run_oja(training_set, learning_rate, epochs_amount):
    oja = OjaPerceptron(training_set, learning_rate)
    oja.train(epochs_amount)
    pass