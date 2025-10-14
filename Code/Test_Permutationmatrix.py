import numpy as np

n=2

# Define the matrix
A = np.array([
    [0, 0.6, 0.7, 0.0001, 0, 0, 0],
    [0.6, 0.2, 0.25, 0, 0, 0, 0],
    [0.4, 0.2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.1, 0.9, 0],
    [0, 0, 0, 0.9999, 0.2, 0, 0],
    [0, 0, 0.05, 0, 0.7, 0, 0.1],
    [0, 0, 0, 0, 0, 0.1, 0.9]
])
A_squared = np.identity(7)
# Compute the square of the matrix
for i in range(n):
    A_squared = A @ A_squared
    
print('\n'.join([''.join(['{:4}'.format(round(item,5))+str("&") for item in row]) 
    for row in A_squared]))

A_multp = A_squared*A_squared[:,2]
print("__________")
print('\n'.join([''.join(['{:4}'.format(round(item,10)) for item in row]) 
    for row in A_multp]))
A_Diff = A_squared-A_multp

print("__________")
print('\n'.join([''.join(['{:4}'.format(round(item,10)) for item in row]) 
    for row in A_Diff]))