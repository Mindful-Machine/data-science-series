import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

"""
General Def:
    - Unsupervised Learning Method often used to:
        * reduce dimensionality of dataset 
        * conserves most of the information
        
Detailed Def:
    - PCA aims to find:
        * new set of ORTHOGONAL dimensions
        * Ranked according to their variance
        * Minimize the projection error
"""

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None #eigenvectors on which we project data points
        self.mean = None #vector means
        self.info = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        
        cov = np.cov(X.T)
        
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        
        print('eigenvalues = ', eigenvalues)
        
        eigenvectors = eigenvectors.T
        
        
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        
        self.info = eigenvalues
        
        self.components = eigenvectors[:self.n_components]
        
    def transform(self, X):
        X  = X - self.mean
        return np.dot(X, self.components.T)   


# data = datasets.load_digits()
data = datasets.load_iris()
X = data.data
y = data.target

# Project the data onto the 2 primary principal components
pca = PCA(2)
pca.fit(X)
X_projected = pca.transform(X)

print("Shape of X:", X.shape)
print("Shape of transformed X:", X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(
    x1, x2, c=y, edgecolor="none", alpha=0.8, cmap=plt.cm.get_cmap("viridis", 3)
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()
plt.show()

#np.cumsum(pca.info/pca.info.sum()) 