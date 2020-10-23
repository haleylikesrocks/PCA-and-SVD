#### Egienfaces compression

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd


# Importing the pictures to use
data = datasets.fetch_olivetti_faces()
images = data.images


# Returns the best rank-k approximation to M
def svd_reconstruct(M, k):
    # TODO: Complete this!
    # Advice: pass in full_matrices=False to svd to avoid dimensionality issues
    u, s, vh = np.linalg.svd(M, full_matrices=False)
    u_new=u[:,:k]
    s_new=s[:k]
    vh_new=vh[:k,:]
    A= u_new @ np.diag(s_new) @ vh_new
    return A

#### Testing to make sure it works
# M=images[1]
# k=3
# A=svd_reconstruct(M, k)
# print(A)
# print(A.shape)

# returns l error
def l_error(M, A):
    error=np.sum(abs(M-A))*1/64*1/64
    return error

#### testing error
# error=l_error(M, A)
# print(error)

# computing the error of the k rank matrix vs the orginial
final_errors=[]
for k in range(1,31):
    k_error=[]
    for M in images:
        A= svd_reconstruct(M, k)
        error=l_error(M, A)
        k_error.append(error)
    final_errors.append(np.mean(k_error))

# plotting up the error
plt.plot(range(1,31), final_errors)
plt.xlabel('k rank')
plt.ylabel('l error')
plt.show()  

# to show the results
M=images[5]
ks=[10,20,30,40]

# showing the orginal
plt.subplot(1,len(ks)+1,1)
plt.title("Original")
plt.axis('off')
plt.imshow(M, cmap='gray')


# Showing the faces with different k ranks
for i in range(len(ks)):
    A=svd_reconstruct(M,ks[i])
    plt.subplot(1,len(ks)+1,i+2)
    plt.title("K="+str(ks[i]))
    plt.axis('off')
    plt.imshow(A, cmap='gray')

plt.show()