import math

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

    return new_transition_matrix, new_emission_matrix

"""
Implements the Baum-Welch algorithm to train HMM parameters (A and B).
"""
def baum_welch(transition_matrix, emission_matrix, pi, observations, max_iterations):
    num_states = len(pi)
    num_emissions = len(emission_matrix[0])
    prev_log_prob = -math.inf

    for _ in range(max_iterations):
        # Forward and backward passes
        alpha, scaling_factors = forward_algorithm(transition_matrix, emission_matrix, pi, observations)
        beta = backward_algorithm(transition_matrix, emission_matrix, observations, scaling_factors)
        
        # Compute gamma and digamma
        gamma, di_gamma = gamma_function(alpha, beta, transition_matrix, emission_matrix, observations)

        # Re estimate parameters
        transition_matrix, emission_matrix = reestimate_parameters(
            gamma, di_gamma, observations, num_states, num_emissions
        )

        # Check for convergence
        log_likelihood = likelihood(scaling_factors)
        if log_likelihood > prev_log_prob:
            prev_log_prob = log_likelihood
        else:
            break

    return transition_matrix, emission_matrix


def format_matrix(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    flat_matrix = [element for row in matrix for element in row]
    return f"{rows} {cols} " + " ".join(f"{x:.6f}" for x in flat_matrix)

def main():
    transition_matrix, _, _ = read_matrix()
    emission_matrix, _, _ = read_matrix()
    pi, _, _ = read_matrix()
    pi = pi[0] # pi as a row vector

    #the sequence of emissions
    last_line = list(map(int, input().strip().split()))
    emissions = last_line[1:]

    A, B = baum_welch(transition_matrix, emission_matrix, pi, emissions, 80)

    # Output transition and emission matrices
    print(format_matrix(A))
    print(format_matrix(B))

if __name__ == "__main__":
    main()