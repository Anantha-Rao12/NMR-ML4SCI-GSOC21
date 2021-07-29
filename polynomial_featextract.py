import numpy as np
import pandas as pd

def split_and_fit(data: np.ndarray, n_split:int,order_fit:int) -> np.ndarray :
    """Given the input numpy 1D array, split it into $(n_split)$ equal halves
    and perform a polynomial fit of order $(order_fit)$ on each half.
    Returns: A new 1D array of size n_split*(order_fit+1) 
    containing the coefficients of the fit for each split"""

    splitted_data = np.array(np.split(data,n_split))
    x_axis = np.arange(len(splitted_data[0])) - len(splitted_data[0])//2
    output_array = np.zeros(n_split*(order_fit+1))

    for counter, row in enumerate(splitted_data):
        output_array[counter*(order_fit +1):
        counter*(order_fit +1)+order_fit+1 ] = np.polyfit(x=x_axis,
                                                          y=row,deg=order_fit)

    return output_array


def precompute_output(dataset:np.ndarray, n_splits:list, order_fits:list) -> np.ndarray:
    """Precomputes the output dataset for the polynomial feature extraction"""

    nelements_ = [i*(j+1) for i,j in zip(n_splits, order_fits)]
    nelements_ = [0] + nelements_
    nelems_cumsum = np.cumsum(nelements_)
    output_data = np.zeros((dataset.shape[0], np.sum(nelements_)))
    return output_data, nelems_cumsum


def poly_featextract_todf(output_data:np.ndarray, n_splits:list, order_fits:list ) -> pd.DataFrame:
    """Given the output of poly_featextract, here we return a dataframe with labelled coulmns"""
    cols = []
    polyfit_features = [f"{i}_{j}" for i in n_splits for j in range(1,i+1)]
    for i,j in zip(n_splits, order_fits):
        for k in polyfit_features:
            if int(k.split("_")[0]) == i:
                cols.extend([f"{k}_{l}" for l in np.arange(j,-1,-1)])
    return pd.DataFrame(output_data,columns=cols)


def poly_featextract(dataset:np.ndarray, n_splits:list, order_fits:list, as_df=False) -> np.ndarray:
    """Computes the polynomial-fit features for each example in the dataset
    Input:
        dataset: 2D array of shape mxn where m is the number of examples and 
                n is the no of features
        n_splits: The number of equal splits features of the dataset 
        order_fits: Order of the polynomial to be fit for each split in n_splits
    Returns:
        2D array with all the polyfit features for the dataset
    """
    if len(order_fits) != len(n_splits):
        raise ValueError("n_splits and order_fits are not equal. Please provide the order of polynomial for each split")

    # precompute the output_dataset for faster execution
    output_data, split_cumsum = precompute_output(dataset,n_splits, order_fits)

    for id_row, row in enumerate(dataset):
        for id_split, split_fit in enumerate(zip(n_splits, order_fits)):
            output_data[id_row,
                        split_cumsum[id_split]:
                        split_cumsum[id_split+1]] = split_and_fit(data=row,
                                                  n_split= split_fit[0],
                                                  order_fit= split_fit[1])
    if as_df == True:
        return poly_featextract_todf(output_data,n_splits, order_fits)
    else:
        return output_data