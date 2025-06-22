
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

def forward_algorithm(transition_matrix, emission_matrix, pi, observations):
    num_states = len(pi)  # Number of states
    T = len(observations)   

    # Initialise alpha
    alpha = [[0] * num_states for _ in range(T)]
    for i in range(num_states):
        alpha[0][i] = pi[i] * emission_matrix[i][observations[0]] #2.8

    #  Induction
    for t in range(1, T):
        for i in range(num_states):
            alpha[t][i] = sum(alpha[t - 1][j] * transition_matrix[j][i] for j in range(num_states)) * emission_matrix[i][observations[t]] #2.13

    # Termination
    p_observations_given_lambda = sum(alpha[T - 1][i] for i in range(num_states)) #2.15
    return p_observations_given_lambda

def main():
    transition_matrix, _, _ = read_matrix()
    emission_matrix, _, _ = read_matrix()
    pi, _, _ = read_matrix()
    pi = pi[0] # pi as a row vector
    
    #the sequence of emissions
    lastLine = list(map(int, input().strip().split()))
    emissions = lastLine[1:] 
    
    probability = forward_algorithm(transition_matrix, emission_matrix, pi, emissions)

    print(f"{probability:.12f}")

if __name__ == "__main__":
    main()


