import os, time
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor

def load_data(datadir_path:str):
    """Given the path of the directory containing the dataset, 
    the respective M_curves data is loaded and returned"""

    readfile = lambda x : pd.read_csv(os.path.join(datadir_path,x),
                                      delimiter=' ',header=None)
    echos_i , echos_r = readfile("echos_i"), readfile("echos_r")
    print("Finished loading rawdata into numpy array")
    return np.abs(echos_i.values + 1j*echos_r.values)

def load_params(datadir_path:str):
    """Given the directory path, loads the input parameter files for the simualtions"""
    
    cols = 'αx αy αz ξ pow Γ3 stencil_type s p d pulse90 pulse180'.split()
    readfile = lambda x : pd.read_csv(os.path.join(datadir_path,x),
                                      delimiter=' ',header=None, 
                                      dtype= np.float32, names=cols)
    print("Finsihed loading parameters file")
    return readfile("echo_params.txt")
    
def load_wlist(datadir_path:str):
    """Given the path of the directory containing the simulation files, 
    load the kernel-integrals file aka "w_list.txt" and return a dataframe"""
    print("finished loading kernel-integrals file.")
    return pd.read_csv(os.path.join(datadir_path,"w_list.txt"), header=None, dtype=np.float64)
                        
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

def get_yclasses(params:pd.DataFrame, ker_integrals:pd.DataFrame) -> pd.DataFrame:
    """Given the params dataframe and the kernel integrals dataframe, here we compute the 
    parameters to be predicted for regression, namely αx, αz, len_scale (ie \sqrt(w_list/(2*αx+αz)))
    Returns: y_classes dataframe"""

    y_classes = params[['αx','αz']].copy()
    y_classes['w_list'] = ker_integrals.values
    def get_len_scale(ax,az,w_list): return np.sqrt((w_list)/(2*ax+az))
    y_classes['len_scale'] = y_classes.apply(lambda row : get_len_scale(row['αx'], row['αz'], row['w_list']), axis=1)
    y_classes.drop('w_list', inplace=True, axis=1)
    return y_classes
    
def fit_RFregressor(X: np.ndarray, y_classes: pd.DataFrame, kfold:int):
    """Fits a RF regressor for all the parameters in y_classes using X and 
    returns the cross-validation scores for X_train and final score on (X_test, y_test)

    Args:
        X : X dataframe that is used to train the model
        y_classes : Pandas dataframe with the predictors-label as the col-name

    Returns: 
        [cv_scores, models, model_scores, y_preds]
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y_classes,
                                                    test_size=0.2, random_state=1,
                                                    stratify=params['stencil_type'] )
    
    cv_scores=[]; models=[]; model_scores=[]; y_preds = [];

    for col in y_train.columns.tolist():
        
        model = RandomForestRegressor(n_estimators=100, max_depth=30,
                              min_samples_split=5, max_features="sqrt",
                             max_samples=0.4, n_jobs=-1)
        
        print(f"Running cross-validation for {col}")
        cv_scores.append(cross_val_score(model, X_train, y_train[col], cv=kfold,
                                         verbose=1, n_jobs=-1))

        print(f"Training for {col}")
        model.fit(X_train, y_train[col])
        models.append(model)
        print(f"Model fitted for {col}. Now scoring")
        model_scores.append(model.score(X_test, y_test[col]))
        y_preds.append((y_test[col], model.predict(X_test)))
        print()

    return [cv_scores, models, model_scores, y_preds]