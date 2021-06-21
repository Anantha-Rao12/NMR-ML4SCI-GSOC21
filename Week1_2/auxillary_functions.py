import os, time
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import PCA

def load_data(path:str, as_df=True):
    """Given the path of the directory containing the data, 
    the respective mat_file and M_curves data is loaded"""
    start_time = time.time()

    M_file_r = "echos_r" # real part of echos
    M_file_i = "echos_i" # imaginary part of echos

    # M(t) curve for each simulation:
    M_r = np.loadtxt(os.path.join(path,M_file_r));
    M_i = np.loadtxt(os.path.join(path,M_file_i));
    print(f"Finished loading data into numpy array. Took {time.time()-start_time:.4}s")
    return np.abs(M_r + 1j*M_i)

def load_params(path:str):
    """Given the directory path loads the input parameter files for the simualtions"""
    params = np.loadtxt(os.path.join(path,"echo_params.txt"))
    cols = 'αx αy αz ξ p Γ3 stencil_type s p d pulse90 pulse180'.split()
    print("Finsihed loading parameters file")
    return pd.DataFrame(params,columns=cols)

def get_window(data:np.ndarray, center_ratio:float, width:float):
    """Returns a subset of the given array with only those datapoints between 
    [center - width , center + width] for all rows/examples"""
    start = int((center_ratio)*data.shape[1])
    return data[:,start-width:start+width], start

def standard_scale(data:np.ndarray):
    """Standardizing the given array with mean=0 and variance=1 column-wise"""
    mean,std = np.mean(data,axis=0), np.std(data,axis=0)
    return (data-mean)/std

def normalize_minmax(data:np.ndarray):
    """Normalizing the data so that all values are within the new range of 0 and 1"""
    return (data - np.min(data,axis=0))/(np.max(data,axis=0) - np.min(data,axis=0))

def pca1D(data:np.ndarray):
    """Returns a 1D PCA version of the given dataset. 
    The given dataset (input) should already be normalised. 
    Returns: 
        Dim-reduced 1D data, variance explained by the one dimension"""
    pca2D = PCA(n_components=1)
    pca2D_data = pca2D.fit_transform(data)
    variance = pca2D.explained_variance_ratio_
    principal_data = pd.DataFrame(data = pca2D_data, columns = ['PC1'])
    return principal_data, variance
    
def pca2D(data:np.ndarray):
    """Returns a 2D PCA version of the given dataset. 
    The given dataset (input) should already be normalised. 
    Returns: 
        Dim-reduced 2D data, variance explained by the two dimensions"""
    pca2D = PCA(n_components=2)
    pca2D_data = pca2D.fit_transform(data)
    variance = pca2D.explained_variance_ratio_
    principal_data = pd.DataFrame(data = pca2D_data, columns = ['PC1', 'PC2'])
    return principal_data, variance

def pca3D(data:np.ndarray):
    """Returns a 3D PCA version of the given dataset. 
    The given dataset (input) should already be normalised. 
    Returns: 
        Dim-reduced 3D data, variance explained by the two dimensions"""
    pca2D = PCA(n_components=3)
    pca2D_data = pca2D.fit_transform(data)
    variance = pca2D.explained_variance_ratio_
    principal_data = pd.DataFrame(data = pca2D_data, columns = ['PC1', 'PC2','PC3'])
    return principal_data, variance

def get_random_points(start,end,no_points):
    """Generates 'no_points' random points between (start, end)"""
    np.random.seed(1)
    return np.random.randint(start,end,no_points)

def plot_cumvar_pca(data,title):
    """Plots the cumulative variance explained by PCA for different 
    number of components"""
    fig,axes = plt.subplots(nrows=2,ncols=2,figsize=(10,8))
    pca = PCA().fit(data)

    axes[0,0].plot(np.cumsum(pca.explained_variance_ratio_),'bx',alpha=0.6)
    axes[0,1].plot(np.cumsum(pca.explained_variance_ratio_)[:200],'bx',alpha=0.6)
    axes[1,0].plot(np.cumsum(pca.explained_variance_ratio_)[:50],'bx',alpha=0.6)
    axes[1,1].plot(np.cumsum(pca.explained_variance_ratio_)[:10],'bx',markersize=8,alpha=0.6)

    for _,ax in np.ndenumerate(axes):
        ax.set_xlabel('Number of Principal components')
    plt.suptitle(f'Cumulative Variance Expalined by PCA ({title})',fontsize=20)
    return axes
