import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """

    # Initialize a random vector of the same size as the matrix
    n = data.shape[0]
    eigenvector = np.random.rand(n)

    for _ in range(num_steps):
        # Perform matrix-vector multiplication A * eigenvector
        matrix_vector_product = np.dot(data, eigenvector)

        # Calculate the dominant eigenvalue estimate
        eigenvalue = float(np.linalg.norm(matrix_vector_product)) 

        # Normalize the eigenvector
        eigenvector = matrix_vector_product / eigenvalue

    return eigenvalue, eigenvector
