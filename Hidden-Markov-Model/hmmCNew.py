import math
import random
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
        row = states[i * cols: (i + 1) * cols]
        matrix.append(row)

    return matrix, rows, cols


"""
Computes the log-likelihood of the observation sequence using scaling factors.
The log-likelihood is negative of the sum of log(scaling_factor).
"""
def likelihood(scaling_factor):
    T = len(scaling_factor)
    log_prob = 0
    
    for i in range(T):
        log_prob += math.log(scaling_factor[i])
    log_prob = -log_prob
    return log_prob

"""
Pseudocode from A Revealing Introduction to Hidden Markov Models. 
Implements the forward algorithm to compute the alpha values and scaling factors.
alpha represents the probability of observation sequences up to time t and ending in state i.
"""
def forward_algorithm(transition_matrix, emission_matrix, pi, observations):
    num_states = len(pi) # Number of states
    T = len(observations)

    # Initialise alpha
    alpha = [[0] * num_states for _ in range(T)]
    scaling_factors = [0] * T

    # Compute alpha_0
    for i in range(num_states):
        alpha[0][i] = pi[i] * emission_matrix[i][observations[0]] #2.8
        scaling_factors[0] += alpha[0][i]

    # Scale alpha_0
    scaling_factors[0] = 1 / scaling_factors[0]
    for i in range(num_states):
        alpha[0][i] *= scaling_factors[0]

    # Compute alpha_t for t >= 1
    for t in range(1, T):
        for i in range(num_states):
            alpha[t][i] = sum(alpha[t - 1][j] * transition_matrix[j][i] for j in range(num_states))  #2.13
            alpha[t][i] *= emission_matrix[i][observations[t]]  #2.13
            scaling_factors[t] += alpha[t][i]
        
        # Scale alpha_t
        scaling_factors[t] = 1 / scaling_factors[t]
        for i in range(num_states):
            alpha[t][i] *= scaling_factors[t]

    return alpha, scaling_factors

"""
Implements the backward algorithm to compute the beta values.
beta represents the probability of observing the remaining sequence from time t+1 given state i at time t.
"""
def backward_algorithm(transition_matrix, emission_matrix, observations, scaling_factors):
    num_states = len(transition_matrix)
    T = len(observations)

    # Initialise beta
    beta = [[0] * num_states for _ in range(T)]

    # Compute beta_T-1
    for i in range(num_states):
        beta[T - 1][i] = scaling_factors[T - 1]

    # Compute beta_t for t < T-1
    for t in range(T - 2, -1, -1):
        for i in range(num_states):
            beta[t][i] = sum(
                transition_matrix[i][j] * emission_matrix[j][observations[t + 1]] * beta[t + 1][j] #2.30
                for j in range(num_states)
            )
            # Scale beta_t
            beta[t][i] *= scaling_factors[t]

    return beta

"""
Computes the gamma and digamma values:
gamma: The probability of being in state i at time t, given the entire observation sequence.
digamma: The probability of transitioning from state i at time t to state j at time t+1.
"""
def gamma_function(alpha, beta, transition_matrix, emission_matrix, observations):
    T = len(observations)
    num_states = len(transition_matrix)

    # Initialize gamma and di_gamma
    gamma = [[0] * num_states for _ in range(T)]
    di_gamma = [[[0] * num_states for _ in range(num_states)] for _ in range(T - 1)]

    for t in range(T - 1):
        for i in range(num_states):
            gamma_sum = 0
            for j in range(num_states):
                di_gamma[t][i][j] = alpha[t][i] * transition_matrix[i][j] * emission_matrix[j][observations[t + 1]] * beta[t + 1][j]
                gamma_sum += di_gamma[t][i][j]
            gamma[t][i] = gamma_sum

    # Probability for last state
    for i in range(num_states):
        gamma[T - 1][i] = alpha[T - 1][i] 

    return gamma, di_gamma

"""
Re-estimates the transition matrix A and emission matrix B using the gamma and digamma values.
"""
def reestimate_parameters(gamma, di_gamma, observations, num_states, num_emissions):
    T = len(observations)

    new_pi = [gamma[0][i] for i in range(num_states)]

    # Update transition matrix A
    new_transition_matrix = [[0] * num_states for _ in range(num_states)]
    for i in range(num_states):
        for j in range(num_states):
            numerator = sum(di_gamma[t][i][j] for t in range(T - 1))
            denominator = sum(gamma[t][i] for t in range(T - 1))
            new_transition_matrix[i][j] = numerator / denominator if denominator != 0 else 0

    # Update emission matrix B
    new_emission_matrix = [[0] * num_emissions for _ in range(num_states)]
    for i in range(num_states):
        for k in range(num_emissions):
            numerator = sum(gamma[t][i] for t in range(T) if observations[t] == k)
            denominator = sum(gamma[t][i] for t in range(T))
            new_emission_matrix[i][k] = numerator / denominator if denominator != 0 else 0

    return new_transition_matrix, new_emission_matrix, new_pi

"""
Implements the Baum-Welch algorithm to train HMM parameters (A and B).
"""
def baum_welch(transition_matrix, emission_matrix, pi, observations, max_iterations):
    num_states = len(pi)
    num_emissions = len(emission_matrix[0])
    prev_log_prob = -math.inf

    for iter in range(max_iterations):
        # Forward and backward passes
        alpha, scaling_factors = forward_algorithm(transition_matrix, emission_matrix, pi, observations)
        beta = backward_algorithm(transition_matrix, emission_matrix, observations, scaling_factors)
        
        # Compute gamma and digamma
        gamma, di_gamma = gamma_function(alpha, beta, transition_matrix, emission_matrix, observations)

        # Re estimate parameters
        transition_matrix, emission_matrix, pi_matrix = reestimate_parameters(
            gamma, di_gamma, observations, num_states, num_emissions
        )

        # Check for convergence
        log_likelihood = likelihood(scaling_factors)
        if abs(log_likelihood - prev_log_prob) < 1e-6:
            break
        else:
            prev_log_prob = log_likelihood

    return transition_matrix, emission_matrix, pi_matrix,  iter+1


def format_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    flat_matrix = [element for row in matrix for element in row]
    return f"{rows} {cols} " + " ".join(f"{x:.6f}" for x in flat_matrix)



def initialize_random_matrix(rows, cols):
    """Generate a matrix with random probabilities where each row sums to 1, rounded to 2 decimals."""
    matrix = []
    for _ in range(rows):
        row = [random.random() for _ in range(cols)]
        row_sum = sum(row)
        normalized_row = [round(x / row_sum, 2) for x in row]  # Normalize and round to 2 decimals
        matrix.append(normalized_row)
    return matrix

def initialize_random_vector(size):
    """Generate a vector with random probabilities that sum to 1, rounded to 2 decimals."""
    vector = [random.random() for _ in range(size)]
    vector_sum = sum(vector)
    normalized_vector = [round(x / vector_sum, 2) for x in vector]  # Normalize and round to 2 decimals
    return normalized_vector

def train_hmm(hidden_states, emissions, max_iterations=1000):
    num_emissions = 4  
    
    # Initialize random matrices
    A_initial = initialize_random_matrix(hidden_states, hidden_states)
    B_initial = initialize_random_matrix(hidden_states, num_emissions)
    pi_initial = initialize_random_vector(hidden_states)

    print("\nINITIAL Transition Matrix A:")
    for row in A_initial:
        print(" ".join(f"{x:.2f}" for x in row))
    
    print("\nINITIAL Emission Matrix B:")
    for row in B_initial:
        print(" ".join(f"{x:.2f}" for x in row))

    print("\nINITIAL Pi Matrix:")
    print(" ".join(f"{x:.2f}" for x in pi_initial))

    # Train using Baum-Welch
    A_trained, B_trained, pi_trained, iterations = baum_welch(A_initial, B_initial, pi_initial, emissions, max_iterations)
    
    # Compute log-likelihood
    alpha, scaling_factors = forward_algorithm(A_trained, B_trained, pi_trained, emissions)
    log_likelihood = likelihood(scaling_factors)
    
    return A_trained, B_trained, pi_trained, log_likelihood, iterations

def q7():
    A_initial, _, _ = read_matrix()
    B_initial, _, _ = read_matrix()
    pi, _, _ = read_matrix()
    pi = pi[0] # pi as a row vector

    #the sequence of emissions
    last_line = list(map(int, input().strip().split()))
    emissions = last_line[1:]

    for ob_size in [1,2,3,4,5,6,7, 100]: # Set to just 1000 for q8
        subset_emissions = emissions[:ob_size] 

        print(f"\nTraining on {ob_size} observations:")
        A_trained, B_trained, pi_trained, iterations = baum_welch(A_initial, B_initial, pi, subset_emissions, 1000)
        
        print(str(iterations) + " iterations")

        print("Trained Transition Matrix:")
        for row in A_trained:
            print(" ".join(f"{x:.6f}" for x in row))

        print("\nTrained Emission Matrix:")
        for row in B_trained:
            print(" ".join(f"{x:.6f}" for x in row))

        print("\nTrained Pi Matrix:")
        print(" ".join(f"{x:.6f}" for x in pi_trained))


def q9():
    # Read emissions sequence
    last_line = list(map(int, input().strip().split()))
    emissions = last_line[1:]

    # Train HMMs with different numbers of hidden states
    for hidden_states in [2, 3, 4, 5]:
        print(f"\nTraining with {hidden_states} hidden states:")
        A_trained, B_trained, pi_trained, log_likelihood, iterations = train_hmm(hidden_states, emissions)
        
        print(f"Log-likelihood: {log_likelihood:.6f}")
        print(f"Converged in {iterations} iterations")
        
        print("\nTrained Transition Matrix (A):")
        for row in A_trained:
            print(" ".join(f"{x:.6f}" for x in row))
        
        print("\nTrained Emission Matrix (B):")
        for row in B_trained:
            print(" ".join(f"{x:.6f}" for x in row))
        
        print("\nTrained Pi Vector:")
        print(" ".join(f"{x:.6f}" for x in pi_trained))

def initialize_uniform_matrix(rows, cols):
    matrix = [[1.0 / cols for _ in range(cols)] for _ in range(rows)]
    return matrix

def initialize_diagonal_matrix(size):
    matrix = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    return matrix

def initialize_pi_with_third_state(size):
    pi = [0.0 for _ in range(size)]
    pi[2] = 1.0  # Start in the third state
    return pi


def initialize_random_matrix(rows, cols):
    matrix = []
    for _ in range(rows):
        row = [random.random() for _ in range(cols)]
        row_sum = sum(row)
        normalized_row = [round(x / row_sum, 2) for x in row]  # Normalize and round to 2 decimals
        matrix.append(normalized_row)
    return matrix

def main():
    # Read emission sequence
    last_line = list(map(int, input().strip().split()))
    emissions = last_line[1:]  

    num_hidden_states = 3
    num_emissions = 4  

    # PART 1: Uniform Initialization
    print("\n--- Uniform Initialization ---")
    A_uniform = initialize_uniform_matrix(num_hidden_states, num_hidden_states)
    B_uniform = initialize_uniform_matrix(num_hidden_states, num_emissions)
    pi_uniform = initialize_uniform_matrix(1, num_hidden_states)[0]

    # Train using Baum-Welch
    A_trained, B_trained, pi_trained, iterations = baum_welch(A_uniform, B_uniform, pi_uniform, emissions, 1000)
    
    # Compute log-likelihood using forward algorithm after training
    alpha, scaling_factors = forward_algorithm(A_trained, B_trained, pi_trained, emissions)
    log_likelihood = likelihood(scaling_factors)
    
    print(f"Log-likelihood: {log_likelihood:.6f}, Iterations: {iterations}")
    print("\nTrained Transition Matrix (A):")
    for row in A_trained:
        print(" ".join(f"{x:.6f}" for x in row))
    print("\nTrained Emission Matrix (B):")
    for row in B_trained:
        print(" ".join(f"{x:.6f}" for x in row))
    print("\nTrained Pi Vector:")
    print(" ".join(f"{x:.6f}" for x in pi_trained))

    # PART 2: Diagonal A and pi = [0, 0, 1]
    print("\n--- Diagonal A and pi = [0, 0, 1] ---")
    A_diagonal = initialize_diagonal_matrix(num_hidden_states)
    B_random = initialize_random_matrix(num_hidden_states, num_emissions) 
    pi_third_state = initialize_pi_with_third_state(num_hidden_states)

    # Train using Baum-Welch
    A_trained, B_trained, pi_trained, iterations = baum_welch(A_diagonal, B_random, pi_third_state, emissions, 1000)
    
    # Compute log-likelihood using forward algorithm after training
    alpha, scaling_factors = forward_algorithm(A_trained, B_trained, pi_trained, emissions)
    log_likelihood = likelihood(scaling_factors)
    
    print(f"Log-likelihood: {log_likelihood:.6f}, Iterations: {iterations}")
    print("\nTrained Transition Matrix (A):")
    for row in A_trained:
        print(" ".join(f"{x:.6f}" for x in row))
    print("\nTrained Emission Matrix (B):")
    for row in B_trained:
        print(" ".join(f"{x:.6f}" for x in row))
    print("\nTrained Pi Vector:")
    print(" ".join(f"{x:.6f}" for x in pi_trained))

    # PART 3: Close to the Solution
    print("\n--- Close to Solution Initialization ---")
    A_solution = [[0.6, 0.07, 0.3], [0.05, 0.75, 0.1], [0.25, 0.35, 0.45]]

    B_solution = [[0.65, 0.25, 0.15, 0.05], [0.15, 0.35, 0.3, 0.25], [0.05, 0.1, 0.2, 0.6]]
    pi_solution = [0.9, 0.1, 0.1]

    # Train using Baum-Welch
    A_trained, B_trained, pi_trained, iterations = baum_welch(A_solution, B_solution, pi_solution, emissions, 1000)
    
    # Compute log-likelihood using forward algorithm after training
    alpha, scaling_factors = forward_algorithm(A_trained, B_trained, pi_trained, emissions)
    log_likelihood = likelihood(scaling_factors)
    
    print(f"Log-likelihood: {log_likelihood:.6f}, Iterations: {iterations}")
    print("\nTrained Transition Matrix (A):")
    for row in A_trained:
        print(" ".join(f"{x:.6f}" for x in row))
    print("\nTrained Emission Matrix (B):")
    for row in B_trained:
        print(" ".join(f"{x:.6f}" for x in row))
    print("\nTrained Pi Vector:")
    print(" ".join(f"{x:.6f}" for x in pi_trained))


if __name__ == "__main__":
    main()