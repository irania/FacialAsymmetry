# Create a feature selection method to select features for classification .
import pandas as pd
from sklearn.decomposition import PCA


def select_features_PCA(X, y,feature_num):
    indexes = X.index
    # Perform PCA
    pca = PCA(n_components=feature_num)
    pca.fit(X)
    X = pca.transform(X)

    # Create a DataFrame of the features
    features = pd.DataFrame(X)

    return features


def select_features_RFE(X,y,feature_num):
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    # Create a model
    model = LogisticRegression()

    # Create the RFE model and select 3 attributes
    rfe = RFE(model, n_features_to_select=feature_num)
    rfe = rfe.fit(X, y)

    return X[X.columns[rfe.support_]], rfe.support_