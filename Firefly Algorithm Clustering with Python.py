# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 13:54:48 2024

@author: S.M.H Mousavi
"""

import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Starting FA Clustering
np.random.seed(0)
random.seed(0)

# Loading
X = pd.read_csv('iris.csv').values

k = 3  # Number of Clusters

import numpy as np

def ClusterCost(m, X):
    # Initialize the distance matrix
    distances = np.zeros((X.shape[0], m.shape[0]))
    
    # Calculate the distance from each point in X to each centroid in m
    for i in range(m.shape[0]):
        distances[:, i] = np.linalg.norm(X - m[i], axis=1)
    
    # Assign Clusters and Find Closest Distances
    ind = np.argmin(distances, axis=1)
    
    # Sum of Within-Cluster Distance
    WCD = np.sum(np.min(distances, axis=1))
    
    return WCD, {'d': distances, 'ind': ind}


# Firefly Algorithm Parameters
MaxIt = 50  # Maximum Number of Iterations
nPop = 30  # Number of Fireflies (Swarm Size)
gamma = 1  # Light Absorption Coefficient
beta0 = 2  # Attraction Coefficient Base Value
alpha = 0.2  # Mutation Coefficient
alpha_damp = 0.98  # Mutation Coefficient Damping Ratio
delta = 0.05 * (np.max(X, axis=0) - np.min(X, axis=0))  # Uniform Mutation Range
m = 2
if np.isscalar(np.min(X)) and np.isscalar(np.max(X)):
    dmax = (np.max(X) - np.min(X)) * math.sqrt(X.shape[1])
else:
    dmax = np.linalg.norm(np.max(X, axis=0) - np.min(X, axis=0))

# Empty Firefly Structure
firefly = {'Position': None, 'Cost': None, 'Out': None}

# Initialize Population Array
pop = np.empty(nPop, dtype=object)

# Initialize Best Solution Ever Found
BestSol = {'Position': None, 'Cost': np.inf}

# Create Initial Fireflies
for i in range(nPop):
    pop[i] = firefly.copy()
    pop[i]['Position'] = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, X.shape[1]))
    pop[i]['Cost'], pop[i]['Out'] = ClusterCost(pop[i]['Position'], X)
    if pop[i]['Cost'] <= BestSol['Cost']:
        BestSol = pop[i].copy()

# Array to Hold Best Cost Values
BestCost = np.zeros(MaxIt)

# Firefly Algorithm Main Loop
for it in range(MaxIt):
    newpop = np.empty(nPop, dtype=object)
    for i in range(nPop):
        newpop[i] = firefly.copy()
        newpop[i]['Cost'] = np.inf
        for j in range(nPop):
            if pop[j]['Cost'] < pop[i]['Cost']:
                rij = np.linalg.norm(pop[i]['Position'] - pop[j]['Position']) / dmax
                beta = beta0 * np.exp(-gamma * rij ** m)
                e = delta * np.random.uniform(-1, 1, (k, X.shape[1]))
                newsol = {}
                newsol['Position'] = pop[i]['Position'] + beta * np.random.uniform(0, 1, (k, X.shape[1])) * (
                            pop[j]['Position'] - pop[i]['Position']) + alpha * e
                newsol['Position'] = np.maximum(newsol['Position'], np.min(X, axis=0))
                newsol['Position'] = np.minimum(newsol['Position'], np.max(X, axis=0))
                newsol['Cost'], newsol['Out'] = ClusterCost(newsol['Position'], X)
                if newsol['Cost'] <= newpop[i]['Cost']:
                    newpop[i] = newsol.copy()
                    if newpop[i]['Cost'] <= BestSol['Cost']:
                        BestSol = newpop[i].copy()
    # Merge
    pop = np.concatenate((pop, newpop))
    # Sort
    pop = sorted(pop, key=lambda x: x['Cost'])
    # Truncate
    pop = pop[:nPop]
    # Store Best Cost Ever Found
    BestCost[it] = BestSol['Cost']
    print('Iteration', it + 1, ': Best Cost =', BestCost[it])

import numpy as np
import matplotlib.pyplot as plt

# Plot for Best Cost Evolution
plt.figure()  # Start a new figure for the Best Cost plot
plt.plot(BestCost, 'k', linewidth=3)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.grid(True)
plt.show()

def PlotRes(X, sol):
    # Start a new figure for the Clustering results
    plt.figure()

    # Cluster Centers
    m = sol['Position']
    
    # Cluster Indices
    ind = sol['Out']['ind']
    
    # Determine the number of unique clusters
    unique_clusters = np.unique(ind)
    num_clusters = len(unique_clusters)

    # Generate colors for each cluster
    Colors = plt.cm.hsv(np.linspace(0, 1, num_clusters))
    
    # Plot each cluster and its center
    for j in range(num_clusters):
        Xj = X[ind == unique_clusters[j], :]
        plt.scatter(Xj[:, 0], Xj[:, 3], s=8, color=Colors[j, :], label=f'Cluster {j+1}')
        plt.scatter(m[j, 0], m[j, 3], s=100, color='black', marker='x')  # Plotting the cluster center

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 4')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
PlotRes(X, BestSol)


# Print FA Results
print("FA Final Cost is :", BestSol['Cost'])

print("Cluster Indices (ind) from the Best Solution:", BestSol['Out']['ind'])

print("Cluster Centers (Position) from the Best Solution:", BestSol['Position'])
