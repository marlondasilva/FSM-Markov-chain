import numpy as np

# Matriz 10x10 de uma Cadeia de Markov Absorvente
P = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Estado 0 (Absorvente)
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Estado 1 (Absorvente)
    [0.2, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # Estado 2 (Transitório)
    [0.0, 0.1, 0.2, 0.4, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0], # Estado 3
    [0.0, 0.0, 0.0, 0.1, 0.6, 0.3, 0.0, 0.0, 0.0, 0.0], # Estado 4
    [0.0, 0.0, 0.0, 0.0, 0.2, 0.5, 0.3, 0.0, 0.0, 0.0], # Estado 5
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.4, 0.2, 0.0, 0.0], # Estado 6
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.7, 0.2, 0.0], # Estado 7
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.2], # Estado 8
    [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.4]  # Estado 9 (Transitório)
]

# Converter para array do NumPy para facilitar cálculos
P = np.array(P)

# ~ A = (P-np.eye(10)).T
# ~ b = np.zeros(10)
# ~ A[-1,:] = np.ones(10)
# ~ b[-1] = 1
Q = P[2:,2:]
R = P[2:, :2]

N = np.linalg.inv(np.eye(8)-Q)
N = np.round(N + 0.0, decimals=2)
t_m = N.sum(axis=1)

B = np.dot(N, R)
print(B)

# ~ x = np.linalg.solve(A,b)
# ~ print(x)
