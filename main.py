import sys
import numpy as np
from random import normalvariate
from math import sqrt
from numpy.linalg import norm
#import plotly.graph_objects as go

#random vector for convergence
def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]


def one_dim_svd(A):
    
    epsilon=1e-10
    n, m = A.shape
    currentV = randomUnitVector(min(n,m))
    lastV = None 

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            return currentV


def SVD(A):
    epsilon = 1e-10
    
    A = np.array(A, dtype=float)
    n, m = A.shape
    k = min(n, m)

    E = np.zeros(k)
    U = np.zeros((n, k))
    V = np.zeros((m, k))

    for i in range(k):
        matrix = A.copy()

        for c in range(i):
            s, u, v = E[c], U.T[c], V.T[c]
            matrix -= s * np.outer(u, v)
            #eliminate last dominant eigenvalue

        if n > m:
            v = one_dim_svd(matrix)  
            u_unnormalized = np.dot(A, v)
            E[i] = norm(u_unnormalized) 
            U[:, i] = u_unnormalized / E[i]
            V[:, i] = v
        else:
            u = one_dim_svd(matrix) 
            v_unnormalized = np.dot(A.T, u)
            E[i] = norm(v_unnormalized)
            U[:, i] = u
            V[:, i] = v_unnormalized / E[i]

    return U, E, V.T


# Q->P
def find_R_t(Q,P):
    
    N = P.shape[0];
    
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    p_centered = P - np.tile(centroid_P, (N, 1))
    q_centered = Q - np.tile(centroid_Q, (N, 1))
    
    p=p_centered.T
    q=q_centered.T
    
    H = np.dot(q,p.T)
    U,E, Vt = SVD(H)
    
    #instead using S matrix
    #only multiply with -1 of the last column of V
    if np.linalg.det(np.dot(U,Vt.T)) <= 0.0:
        Vt.T[-1, :] *= -1.0

    # Optimal rotation
    R = np.dot(Vt.T, U.T)
    
    t = centroid_P - np.dot(R,centroid_Q) 

    return R, t


def main():
    mat1_file = sys.argv[1]
    mat2_file = sys.argv[2]
    correspondences_file = sys.argv[3]
    
    Q_all = np.genfromtxt(mat1_file)
    P_all = np.genfromtxt(mat2_file)
    correspondences = np.genfromtxt(correspondences_file)
    correspondences = correspondences.astype(int)

    #Choose the corresponding points
    Q = Q_all[correspondences[:, 0]]
    P = P_all[correspondences[:, 1]]

    R, t = find_R_t(Q,P)
    
    #Delete corresponding points
    noncorresponded_P = np.delete(P_all, correspondences[:, 1], axis=0)

    rotated_Pt = np.dot(R.T, np.subtract(noncorresponded_P.T,t.reshape(-1, 1)))
    
    #Merge two dataset
    merged_rows = []
    max_rows = max(len(Q_all), len(rotated_Pt.T))
    for i in range(max_rows):
        if i < len(rotated_Pt.T):
            merged_rows.append(rotated_Pt.T[i])
        if i < len(Q_all):
            merged_rows.append(Q_all[i])

    merged_data = np.array(merged_rows)
    np.savetxt('merged.txt', merged_data, fmt='%.18e')
    np.savetxt('translation_vec.txt',t.T ,fmt='%.18e')
    np.savetxt('rotation_mat.txt', R ,fmt='%.18e')

if __name__ == "__main__":
    main()

#------------Visualization--------------------

'''
x = merged_data[:, 0]
y = merged_data[:, 1]
z = merged_data[:, 2]



# 3D scatter plot 
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color='red',   # point's color
        opacity=0.2
    )
)])

fig.update_layout(scene=dict(
    xaxis_title='X Axis',
    yaxis_title='Y Axis',
    zaxis_title='Z Axis'
))

# The image
fig.show()
'''
