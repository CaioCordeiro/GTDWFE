import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from scipy.spatial.distance import jaccard
from sklearn.metrics import mutual_info_score
import itertools as it
import math
import matplotlib.pyplot as plt

def match(a, b):
    return [ b.index(x) if x in b else None for x in a ]

# takes three random variables as input and computes the conditional mutual information in nats according to the entropy estimator
def cmi(x, y, z):
    # first compute the joint distribution
    xy = np.histogram2d(x, y, bins=5, normed=True)[0]
    xz = np.histogram2d(x, z, bins=5, normed=True)[0]
    yz = np.histogram2d(y, z, bins=5, normed=True)[0]

    # then compute the conditional distributions
    pxy = xy / np.sum(xy)
    pxz = xz / np.sum(xz)
    pyz = yz / np.sum(yz)

    # finally compute the conditional mutual information
    Hxy = -np.sum(pxy * np.log2(pxy))
    Hxz = -np.sum(pxz * np.log2(pxz))
    Hyz = -np.sum(pyz * np.log2(pyz))

    MI = Hxz + Hyz - Hxy

    return MI

def GT_feature_selection(df, X, y, n):
    columns_arr = X.columns
    features_size = len(columns_arr)
    p = 4
    # Number of selected features
    Th = n
    #weight
    w = [1 for i in range(features_size)]
    sum_RR = [0 for i in range(features_size)]
    lf = [0 for i in range(features_size)]
    flag = [0 for i in range(features_size)]
    Banzhaf_power = [0 for i in range(features_size)]
    list_z = []
    col_added = []
    t=1
    CMI=0
    MI=0
    # Calculate Pearson's correlation coefficient and Tanimoto coefficient
    for i in range(len(columns_arr)):
        summation = 0
        corre = abs(np.corrcoef(df[columns_arr[i]], y.to_numpy())[0][1])
        for j in range(len(columns_arr)):
            if i != j:
                df_copy = X.copy(deep=True)
                feature_1 = df_copy[columns_arr[i]]
                feature_2 = df_copy[columns_arr[j]]
                tanimoto_coeff = jaccard(feature_1, feature_2)
                summation = summation+tanimoto_coeff
        tanimoto_coeff_avg = summation/(features_size-1)
        sum_RR[i]= corre+tanimoto_coeff_avg 
    for t in range(Th):
        for i, feature in enumerate(X.columns):
            if flag[i] != 1:
                lf[i] = sum_RR[i]*w[i]
        # Select feature with largest lf
        maximum = 0
        index = 0
        for i in range(features_size):
            if flag[i] != 1:
                if lf[i] > maximum:
                    maximum = lf[i]
                    index = i
        flag[index] = 1
        list_z.append(df[columns_arr[index]])
        col_added.append(columns_arr[index])

        len_col = len(col_added)
        # Calculate Banzhaf power index
        for i in range(features_size):
            if flag[i] != 1:
                combin = 0
                for y_ in range(p):
                    if len_col >= y_:
                        aa = list(it.combinations(col_added, y_))
                        len_ = len(aa)
                        combin = combin + len_
                        for g in range(len_):
                            if not set(aa[g]).isdisjoint(set(columns_arr[index])):
                                h = aa[g]
                                count = 0
                                sum_col_MI = 0
                                for q in range(y_):
                                    col_posit = match(h[q], columns_arr)
                                    sum_col_MI= sum_col_MI + cmi(df[:col_posit], y.to_numpy(), df[i+1:])-mutual_info_score(df[:col_posit], y.to_numpy())
                                sum_col_MI = sum_col_MI/y_
                                for v in range(y_):
                                    posit = match(h[v], columns_arr)
                                    CMI = cmi(df[:posit], y.to_numpy(), df[i+1:])
                                    MI = mutual_info_score(df[:posit], y.to_numpy())
                                    if CMI > MI:
                                        count = count + 1
                                if sum_col_MI >= 0 and count >= math.ceil(y_/2):
                                    Banzhaf_power[i] = Banzhaf_power[i] + 1
                w[i] = w[i]+Banzhaf_power[i]/combin

    return pd.DataFrame(list_z).T
    
                                
                                                         

