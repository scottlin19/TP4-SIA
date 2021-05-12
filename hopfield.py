import numpy as np

# class HopfieldNeuron:
#     def __init__(self,weights):
#         self.weights = weights
#         self.state = 0

#     def activation(self,prev_state):
#         pass
    
#     def __repr__(self):
#         return f"Neuron - w: {self.weights}\n"
    
#     def __str__(self):
#         return f"Neuron - w: {self.weights}\n" 

class Hopfield: 
    
    def __init__(self, stored_patterns):
        self.stored_patterns = np.array(stored_patterns) #[ [ 1,2,3,... 25], [1,2 ..., 25], ... 4 letters ]
        self.pattern_dimension = len(stored_patterns[0])
        # self.w = np.zeros((self.pattern_dimension,self.pattern_dimension), dtype=float)
 
        self.w = np.dot(self.stored_patterns.T, self.stored_patterns) / self.pattern_dimension

        # for i in range(self.pattern_dimension):
        #     for j in range(self.pattern_dimension):
        #         if i == j: 
        #             w[i][j] = 0  
        np.fill_diagonal(self.w, 0)
        
    def train(self,unknown_pattern):
        states = np.array(unknown_pattern)
        prev_states = np.zeros((25,), dtype=int)
        # Iterate until a steady state is reached
        while not np.array_equal(states, prev_states):
            print('while')
            prev_states = states
            states = self.activations(states)
        
        # found pattern 
        for stored in self.stored_patterns:
            print(np.array_equal(states, stored))
            print(stored)
            print(states)
            if np.array_equal(states, stored):
                return (True,stored)    # associated pattern 
        return (False,states)           # spurious state  

    def activations(self, states):
        
        ret = []
        for i in range(self.pattern_dimension):
            excited_state = np.inner(self.w[i], states)   #s[i] = w[i][j] * s[j]
            print("EXCITED")
            print(excited_state)
            if( excited_state != 0): 
                print(f"was: {states[i]}, is: {self.step(excited_state)}")
                ret.append(self.step(excited_state))
            else: 
                ret.append(states[i]) # if h = 0 previous state 
        print("ret")
        print(ret)
        return np.array(ret)

    def step(self,n):
        if(n > 0): 
            return 1
        elif n < 0: 
            return -1

def run_hopfield(stored_patterns, unknown_pattern): 
    hopfield = Hopfield(stored_patterns)
    found, pattern = hopfield.train(unknown_pattern) 
    if found: 
        print(f"Pattern was found:\n{pattern}")
    else:
        print(f"Pattern NOT found, final pattern:\n{pattern}")
    
        