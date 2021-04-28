import sklearn_relief as relief

"implements the Relieff feature selection algorithm"
def Relieff(X, y, n):
    return relief.RReliefF( n_features=n, k = 5 ).fit_transform(X, y)
