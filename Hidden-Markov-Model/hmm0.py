def read_matrix():
    # Read number of rows, columns and the states. 
    tokens = list(map(float, input().strip().split()))
    rows = int(tokens[0])
    cols = int(tokens[1])
    states = tokens[2:]
    
    if len(states) != rows * cols:
        raise ValueError("Number of elements does not match the dimensions!!")
    
    matrix = []
    for i in range(rows):
        # extract all states for row i
        row = states[i*cols : (i+1)*cols] #start and end index
        matrix.append(row)
    
    return matrix, rows, cols

def matrix_multiplication(matrix1, rows1, cols1, matrix2, rows2, cols2):
    if cols1 != rows2:
        raise ValueError("Matrix dimensions do not match!")

    new_matrix = [[0] * cols2 for _ in range(rows1)] #dimensions of new matrix

    # Calculate dot product 
    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                new_matrix[i][j] += matrix1[i][k] * matrix2[k][j]

    return new_matrix, rows1, cols2

def main():
    transition_matrix, t_rows, t_cols = read_matrix()
    emission_matrix, e_rows, e_cols = read_matrix()
    initial_state_distribution, i_rows, i_cols = read_matrix()
    
    next_state_distribution, n_rows, n_cols = matrix_multiplication(initial_state_distribution, i_rows, i_cols, transition_matrix, t_rows, t_cols)
    emission_probabilities, em_rows, em_cols = matrix_multiplication(next_state_distribution, n_rows, n_cols, emission_matrix, e_rows, e_cols)
    
    print(f"{em_rows} {em_cols} ", end="")
    for row in emission_probabilities:
        print(" ".join(map(str, row)), end=" ")
    
if __name__ == "__main__":
    main()