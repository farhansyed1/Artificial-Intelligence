# A: transisition matrix
# B: emission matrix
# o: observations
# delta: maximum prob of being in certain state matrix
# delta idx: index of state for max prob


def read_matrix():
    tokens = list(map(float, input().strip().split()))
    rows = int(tokens[0])
    cols = int(tokens[1])
    states = tokens[2:]
    
    if len(states) != rows * cols:
        raise ValueError("Number of elements does not match the dimensions!!")
    
    matrix = []
    for i in range(rows):
        #extract all states for row i
        row = states[i*cols : (i+1)*cols] #start and end index
        matrix.append(row)
    
    return matrix, rows, cols

def viterbi_algorithm(transition_matrix, emission_matrix, pi, observations):
    num_states = len(pi)  # Number of states
    T = len(observations)  

    # Initialise delta 
    delta = [[0] * num_states for _ in range(T)]  
    delta_idx = [[0] * num_states for _ in range(T)]   

    for i in range(num_states):
        delta[0][i] = pi[i] * emission_matrix[i][observations[0]] # 2.17
        delta_idx[0][i] = 0  

    # Recursion
    #finding the max probability of being in state i at time t, 
    for t in range(1, T):
        for i in range(num_states): 
            # by observing all previous probability of being in state i at time t (delta) * transition probability from state j to state i. 
            # max_state: the index of the maximum probability
            max_probability, max_state = max(
                (delta[t - 1][j] * transition_matrix[j][i], j) for j in range(num_states)
            ) # max [delta (j) * a_ji]
           
            delta[t][i] = max_probability * emission_matrix[i][observations[t]] #2.19
            delta_idx[t][i] = max_state # 2.20
            #Stores the index j of the previous state that maximized the probability for transitioning to state i.

    # Termination
    max_probability, last_state = max((delta[T - 1][i], i) for i in range(num_states)) #2.21

    # Step 4: Path Backtracking
    path = [0] * T
    path[T - 1] = last_state # from 2.21 above - Starts from last_state (the most likely state at time T)
    for t in range(T - 2, -1, -1):
        path[t] = delta_idx[t + 1][path[t + 1]] #2.23 - Traces backward through delta_idx to reconstruct the entire state sequence

    return path

def main():
    transition_matrix, _, _ = read_matrix()
    emission_matrix, _, _ = read_matrix()
    pi, _, _ = read_matrix()
    pi = pi[0] # pi as a row vector
    
    #the sequence of emissions
    lastLine = list(map(int, input().strip().split()))
    emissions = lastLine[1:] 
    
    most_likely_states = viterbi_algorithm(transition_matrix, emission_matrix, pi, emissions)

    print(" ".join(map(str, most_likely_states)))

if __name__ == "__main__":
    main()


