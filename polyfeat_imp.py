import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score


def get_fi(model, X_data:pd.DataFrame, kind:str):
    """Get feature importance dataframe from fitted model"""

    if kind == 'timeseries':
        fi = pd.DataFrame(np.array([X_data.columns.tolist(),
                                    model.feature_importances_]).T,
             columns=['feature','fi'])
        fi['fi'] = pd.to_numeric(fi['fi'])
        return fi.sort_values('fi', ascending=False)

    if kind == "polyfeatures":
        fi = pd.DataFrame(np.array([X_data.columns.tolist(),
                       model.feature_importances_]).T,
             columns=['feature','fi'])
        fi["fi"] = pd.to_numeric(fi["fi"], downcast="float")
        fi['order'] = fi['feature'].apply(lambda x : x.split("_")[-1])
        fi["color"] = fi["order"].replace({'0':'r','1':'b','2':'g','3':'y'})

        return fi.sort_values('fi', ascending=False)

def features_vs_r2(X: pd.DataFrame, X_kind:str, y: pd.DataFrame, model, test_size:float,
                   cv:int, niters:int, stratify):
    """Computes the cv score (r2) and best-fit model along with fi importances for the given (X,y). 
    Then the number of features is reduced using feature_imp > median and a new model is trained. 
    Returns a dataframe with full report
    
    Returns:
            [len(to_keep), to_keep, model, new_features.fi, 
                     [scores.mean(), scores.std()], oob_score, test_score]"""
    
    to_keep = X.columns.tolist()
    data= []
    for i in range(niters):
        print(f"Running iter {i+1}")
        m = clone(model)
        print(f"Number of features: {len(to_keep)}")
        X_train, X_test, y_train, y_test = train_test_split(X[to_keep], y, test_size=test_size,
                                                            random_state=101, stratify=stratify)
        print(f"Size of training and test set: {X_train.shape, X_test.shape}")
        print(f"Computing cross-validation scores")
        scores = cross_val_score(m, X_train, y_train, cv=cv, n_jobs=-1)
        print(f"Cross val-scores: {np.mean(scores)  :.3} +- {np.std(scores) :.3}")
        
        print(f"Done! \n Fitting model")
        m.fit(X_train, y_train)
        
        oob_score = m.oob_score_
        print(f"Model Out-of-Bag score: {oob_score:.3}")

        test_score = m.score(X_test, y_test)
        print(f"Test score: {test_score :.3}")
        
        fi = get_fi(m, X_train, X_kind)
        
        data.append([len(to_keep), to_keep, m, fi.fi, 
                     [scores.mean(), scores.std()], oob_score, test_score])
        new_features = fi[fi.fi >= fi.fi.median()]
        to_keep = fi[fi.fi >= fi.fi.median()].feature
        print()
                     
    return data
    
    
def features_vs_f1(X: pd.DataFrame, X_kind:str,  y: np.ndarray, model, test_size:float,
                   cv:int, niters:int, stratify):
    """Computes the cv score (f1) and best-fit model along with fi importances for the given (X,y). 
    Then the number of features is reduced using feature_imp > median and a new model is trained. 
    Returns a dataframe with full report
    
    Returns:
            [len(to_keep), to_keep, model, new_features.fi, 
                     [scores.mean(), scores.std()], oob_score, test_score]"""
    
    to_keep = X.columns.tolist()
    data= []
    for i in range(niters):
        print(f"Running iter {i+1}")
        m = clone(model)
        print(f"Number of features: {len(to_keep)}")
        X_train, X_test, y_train, y_test = train_test_split(X[to_keep], y, test_size=test_size,
                                                            random_state=101, stratify=stratify)
        print(f"Size of training and test set: {X_train.shape, X_test.shape}")
        print(f"Computing cross-validation scores")
        scores = cross_val_score(m, X_train, y_train, cv=cv, scoring='f1_weighted', n_jobs=-1)
        print(f"Cross val-scores: {np.mean(scores)  :.3} +- {np.std(scores) :.3}")
        
        print(f"Done! \n Fitting model")
        m.fit(X_train, y_train)
        
        oob_score = m.oob_score_
        print(f"Model Out-of-Bag score: {oob_score:.3}")

        test_score = m.score(X_test, y_test)
        print(f"Test score: {test_score :.3}")
        
        fi = get_fi(m, X_train, X_kind)
        
        data.append([len(to_keep), to_keep, m, fi.fi, 
                     [scores.mean(), scores.std()], oob_score, test_score])
        new_features = fi[fi.fi >= fi.fi.median()]
        to_keep = fi[fi.fi >= fi.fi.median()].feature
        print()
                     
    return data


class FeaturePlot:
    def __init__(self, ntseries):
        self.ntseries = ntseries

    def get_intervals(self, n_splits:list) -> np.ndarray :
        """Gives the time-interval splits for the number of splits given
        Returns : Dictionary with the n_split as the key and the time-interval stamps as values"""
        split_index = [np.linspace(0, self.ntseries, i+1).astype(np.int32) for i in n_splits]
        intervals = {}
        for idx, interval in zip([4,5,10],split_index):
            intervals[idx] = interval 
        return intervals

def plot_feature(axes, intervals:np.ndarray, feature:tuple, plot_index:int):
    """Given an axes object, plots a line-bar for the given feature of the form (10,5,2) where:
        10 : order of partition 
        5 : the bin of interest in the parition
        2 : order of the polynomial in the given bin
        We choose a color for the order of the polynomial (0th, 1st, 2nd, 3rd)
    Returns : an axes object
    """
    color_arr = ['#c70a39','#698bfa','#90e388','y']

    axes.plot(intervals[int(feature[0])][int(feature[1])-1: int(feature[1])+1],
            [plot_index, plot_index], lw=10,
            color = color_arr[int(feature[2])], label=f"$x^{int(feature[2])}$"
            )
    end_time = [max(intervals[key]) for key in intervals.keys()][0]
    axes.plot([0,end_time],[plot_index+0.5, plot_index+0.5],color=(0,0,0,.2))
    axes.legend()

def fi_df2plot(fi_df: pd.DataFrame, ntop:int, intervals, ax):
    """Plots the feature importance plot based on imp_feats data"""

    feats = fi_df.head(ntop)['feature'].values.tolist()[::-1]
    feats = list(map(lambda x  : x.split('_') , feats))

    ax.set(xlabel="Time")
    for idx, feature in enumerate(feats):
        plot_feature(ax, intervals, feature, idx+1)
    return ax

