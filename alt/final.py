import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy
from scipy import ndimage

from scipy.ndimage import interpolation
from scipy import ndimage as ndi
from skimage import feature

import time
from sklearn.decomposition import PCA # Principal Component Analysis module



X_train = pd.read_csv('input/trainData.csv')
y_train = pd.read_csv('input/trainLabels.csv')
X_test = pd.read_csv('input/kaggleTestSubset.csv')
y_test = pd.read_csv('input/kaggleTestSubsetLabels.csv')



def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix

def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


def preprocess_X(X):
    result = []
    values = X.values
    for v in values:
        curr = v.reshape((28, 28))
        curr = deskew(curr)
        result.append(curr.flatten())
        
    return result

def sobel(X):
    sobel = []
    for im in range(len(X)):
        image = X[im].reshape((28,28))
        sobel.append(ndimage.sobel(image).reshape(784))
#         image = np.array(X.iloc[im]).reshape((28,28))
#         image = ndimage.sobel(image).flatten()
        
#         sobel.append(image)

    return sobel

def canny_process(X):
    canny = []
    for im in range(X.shape[0]):
#         image = im.reshape((28,28))
#         sobel.append(ndimage.sobel(image).reshape(784))
        image = np.array(X.iloc[im]).reshape((28,28))
        image = feature.canny(image, sigma=3).flatten()
        
        canny.append(image)

    return canny


# scale the test data
def reduce_dimensions(X, r_comp = 200):
    pca = PCA(n_components=r_comp)
    X_scaled = StandardScaler().fit_transform(X)
#     print(len(X_scaled[0]))
    mean_vec = np.mean(X, axis=0)
    cov_mat = np.cov(X.T)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Create a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [ (np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]
    # Sort from high to low
    eig_pairs.sort(key = lambda x: x[0], reverse= True)
    # Calculation of Explained Variance from the eigenvalues
    tot = sum(eig_vals)
    var_exp = [(i/tot)*100 for i in sorted(eig_vals, reverse=True)] # Individual explained variance
    cum_var_exp = np.cumsum(var_exp) # Cumulative explained variance
    x_newdim = pca.fit_transform(X_scaled)
    return x_newdim




accuracy = []
def neighbour_accuracy(X_train, y_train, X_test, y_test, initial_i = 1, final_i = 10):
    for i in range(initial_i, final_i):
        knn_model = KNeighborsClassifier(algorithm='auto', n_neighbors=i, metric='minkowski', p=2, n_jobs=-1, weights='distance')
        knn_model.fit(X=X_train, y=y_train.Label)
        
        pred_time = time.time()
        pred = knn_model.predict(X_test)
        pred_time = time.time() - pred_time
        
        _ = accuracy_score(y_test, pred)
        print("For k = " + str(i) + " Accuracy was: " + str(_) + ", Time Taken: " + str(pred_time))
        accuracy.append(_)
    
neighbour_accuracy(X_train/255, y_train, X_test/255, y_test)
x = range(1, 10)
plt.plot(x, accuracy)
plt.title("KNeighbours vs Accuracy")
plt.xlabel('KNeighbours')
plt.ylabel('Accuracy')


deskewed_train = preprocess_X(X_train/255)
deskewed_test = preprocess_X(X_test/255)
knn_model_desk = KNeighborsClassifier(algorithm='auto', leaf_size=30, n_neighbors=6, metric='minkowski', p=3, n_jobs=-1, weights='distance')
knn_model_desk.fit(deskewed_train, y_train.Label)
pred_desk = knn_model_desk.predict(deskewed_test)
print("Deskewed images: " + str(accuracy_score(pred_desk, y_test)))


def checkcomponents():
    print("Decision Tree Accuracies: ")
    for i in range(50, 500, 50):
        X_train_red = reduce_dimensions(X_train, i)
        X_test_red = reduce_dimensions(X_test, i)
        acc = []
        print("Dimensions= " + str(i) + " ", end='')
        for k in range(1, i, 50):
            model = DecisionTreeClassifier(max_depth=k, criterion = "entropy")
            model.fit(X_train_red, y_train)
            acc.append(accuracy_score(y_test, model.predict(X_test_red)))
        print(np.max(acc))
checkcomponents()