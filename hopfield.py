import numpy as np

class Hopfield: 
    
    def __init__(self, stored_patterns):
        self.stored_patterns = np.array(stored_patterns) #[ [ 1,2,3,... 25], [1,2 ..., 25], ... 4 letters ]
        self.pattern_dimension = len(stored_patterns[0])
 
        self.w = np.dot(self.stored_patterns.T, self.stored_patterns) / self.pattern_dimension
        np.fill_diagonal(self.w, 0)
        
    def train(self,unknown_pattern):
        states = np.array(unknown_pattern)
        pretty_print(states,5)
        prev_states = np.zeros((25,), dtype=int)
        # Iterate until a steady state is reached
        while not np.array_equal(states, prev_states):
         
            prev_states = states
            states = self.activations(states)
            pretty_print(states,5)
            print(f"Steady State: {np.array_equal(states, prev_states)}")
        
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

def run_hopfield(stored_patterns, unknown_pattern): 
    hopfield = Hopfield(stored_patterns)
    found, pattern = hopfield.train(unknown_pattern) 
    if found: 
        print(f"Pattern was found:\n{pattern}")
    else:
        print(f"Pattern NOT found, final pattern:\n{pattern}")
    
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