import math
import random
import numpy as np
import matplotlib.pyplot as plt

class OjaPerceptron:

    def __init__(self, training_set, learning_rate):
        self.training_set = np.array(training_set) 
        self.learning_rate = learning_rate
    
    def train(self, epochs_amount):
        registers_amount = len(self.training_set.T[0])
        dimension = len(self.training_set[0])  # ((area1,b,c,d,e,f,g) , (area2,2,3,4,5,6,7) ) 
        w = np.random.uniform(0, 1, dimension) # array de longitud p+1 con valores random entre 0 y 1  
    
        for epochs in range(epochs_amount):
                
            for input in self.training_set: # iterate by rows 
               
                y = np.inner(input, w)      # inner product: sum(x*w) 
                
                delta_w = self.learning_rate * y * (input - y*w) # eta* (y*x - y^2 * w ) = eta * y(x - yw)

                w += delta_w 

        norm = math.sqrt(np.inner(w,w))

        return w / norm  # PC1


def run_oja(training_set, learning_rate, epochs_amount,countries):
    oja = OjaPerceptron(training_set, learning_rate)
    pca1 = oja.train(epochs_amount)
    print(f"Oja eigenvector that builds PC1:\n {pca1}")
    countries_pca1 = [np.inner(pca1,training_set[i]) for i in range(len(training_set))]
    libray_pca1 = [0.12487390183337656,-0.5005058583604993,0.4065181548118897,-0.4828733253002008,0.18811161613179747,-0.475703553912758,0.27165582007504635]
    countries_library_pca1 = [np.inner(libray_pca1,training_set[i]) for i in range(len(training_set))]
    fig,(ax1,ax2) = plt.subplots(1,2)
    bar1 = ax1.bar(countries,countries_pca1)
    bar2 = ax2.bar(countries,countries_library_pca1)
    ax1.set_ylabel('PCA1')
    ax1.set_title('PCA1 per country using Oja')
    ax2.set_ylabel('PCA1')
    ax2.set_title('PCA1 per country using Sklearn')
    ax1.set_xticks(range(len(countries)))
    ax2.set_xticks(range(len(countries)))
    ax1.set_xticklabels(countries, rotation=90)
    ax2.set_xticklabels(countries, rotation=90)
    plt.show()