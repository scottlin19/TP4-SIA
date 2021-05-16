import numpy as np
import matplotlib.pyplot as plt

class Hopfield: 
    
    def __init__(self, stored_patterns):
        self.stored_patterns = np.array(stored_patterns) #[ [ 1,2,3,... 25], [1,2 ..., 25], ... 4 letters ]
        self.pattern_dimension = len(stored_patterns[0])
 
        self.w = np.dot(self.stored_patterns.T, self.stored_patterns) / self.pattern_dimension
        np.fill_diagonal(self.w, 0)
        
    def train(self,unknown_pattern,max_iterations):
        states = np.array(unknown_pattern)
        pretty_print(states,5)
        prev_states = np.zeros((25,), dtype=int)
        energies = []
        energies.append(self.energy_function(states))
        print(f"Energy: {energies[-1]}")
        # Iterate until a steady state is reached
        i = 0
        while i < max_iterations and not np.array_equal(states, prev_states):
            
            prev_states = states
            states = self.activations(states)
            energies.append(self.energy_function(states))
            pretty_print(states,5) 
            print(f"Energy: {energies[-1]}")
            print(f"Steady State: {np.array_equal(states, prev_states)}")
            i += 1
        
        plt.plot([i for i in range(len(energies))],energies)
        plt.ylabel('Energy level')
        plt.xlabel('Iterations')
        # found pattern 
        for stored in self.stored_patterns:
            if np.array_equal(states, stored):
                return (True,states)    # associated pattern 
        return (False,states)           # spurious state  

    def activations(self, states):
        
        ret = []
        for i in range(self.pattern_dimension):
            excited_state = np.inner(self.w[i], states)   #s[i] = w[i][j] * s[j]
       
            if( excited_state != 0): 
                # print(f"was: {states[i]}, is: {self.step(excited_state)}")
                ret.append(self.step(excited_state))
            else: 
                ret.append(states[i]) # if h = 0 previous state 
        return np.array(ret)

    def step(self,n):
        if(n > 0): 
            return 1
        elif n < 0: 
            return -1
        
    def energy_function(self, states):
        h = 0
        for i in range(self.pattern_dimension):
            for j in range(i+1,self.pattern_dimension):
                h += self.w[i][j] * states[i] * states[j]
        return -h
         

def run_hopfield(stored_patterns, unknown_pattern,max_iterations): 
    hopfield = Hopfield(stored_patterns)
    found, pattern = hopfield.train(unknown_pattern,max_iterations) 
    if found: 
        print(f"Pattern was found:\n{pattern}")
        # return found,stored_patterns.index(pattern.tolist())
    else:
        print(f"Pattern NOT found, final pattern:\n{pattern}")
    # return found,-1
    plt.show()
    
    
def pretty_print(pattern,length):
    print(f"Current state: {pattern}")
    mat = pattern.reshape(length,length)
    # print(mat)
    fmt = "".join(["{}" for i in range(length)])
    def pattern_mapper(x):
        if(x == 1):
            return "*"
        return " "
    print("-------------------------------\n")
    for i in range(length):
        print(fmt.format(*list(map(lambda x: pattern_mapper(x),mat[i]))))