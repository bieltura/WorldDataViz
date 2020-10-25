import numpy as np

def get_adjacency_matrix(nodes, connections):
   
    identity = np.identity(nodes)

    adjacency_matrix = identity.copy()
    for node in range(0,nodes):
        for connection in connections[node]:
            adjacency_matrix[node] = adjacency_matrix[node] + identity[connection-1]

    return adjacency_matrix

def weight_graph(adjacency_matrix, node, weight_factor=2):
    matrix = adjacency_matrix.copy()
    matrix[node-1,:] = matrix[node-1,:]*weight_factor
    matrix[:,node-1] = matrix[:,node-1]*weight_factor
    return matrix/matrix.max()