from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import pandas as pd


# Create a classifier: a support vector classifier
classifiers = {
    'SVC': SVC(),
    'KNeighborsClassifier': KNeighborsClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
#    'XGBClassifier': XGBClassifier( eval_metric='mlogloss')
}

param_grids = {
    'SVC': {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 2,10]},
    'KNeighborsClassifier': {'n_neighbors': [3, 5, 7]},
    'RandomForestClassifier': {'n_estimators': [50, 100, 200]},
    'LogisticRegression': {'C': [0.1, 1, 10]},
    'XGBClassifier': {'learning_rate': [0.01, 0.1, 0.2, 1]}
}

def gridsearch_classifiers(X,y):
    # List to save result
    result = []
    best_classifier = None
    for classifier_name, classifier in classifiers.items():
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=param_grids[classifier_name],
                                   scoring='accuracy',
                                   cv=6,
                                   n_jobs=-1)

        grid_search.fit(X, y)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_

        result.append({
            'classifier': classifier_name,
            'best_parameters': best_parameters,
            'best_accuracy': best_accuracy,
        })

        # classification_results(clf, X_test,y_test,y_pred)
    # Convert list to DataFrame
    df = pd.DataFrame(result)
    print(df)




from sklearn.model_selection import cross_val_score

def gridsearch_classifiers_group(X, y, groups):
    result = []
    
    group_kfold = GroupKFold(n_splits=5)
    
    for classifier_name, classifier in classifiers.items():
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=param_grids[classifier_name],
                                   scoring='accuracy',
                                   cv=group_kfold.split(X, y, groups),
                                   n_jobs=-1)
        
        grid_search.fit(X, y)
        best_accuracy = grid_search.best_score_
        best_parameters = grid_search.best_params_
        
        # Get accuracy for each fold
        accuracies = cross_val_score(grid_search.best_estimator_, X, y, groups=groups, cv=group_kfold.split(X, y, groups), scoring='accuracy')

        result.append({
            'classifier': classifier_name,
            'best_parameters': best_parameters,
            'best_accuracy': best_accuracy,
            'fold_accuracies': accuracies  # Adding fold accuracies here
        })

    df = pd.DataFrame(result)
    print(df)
    return df



def gridsearch_classifiers_group_majority(X, y, groups):
    result = []
    group_kfold = GroupKFold(n_splits=5)
    
    for classifier_name, classifier in classifiers.items():
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=param_grids[classifier_name],
                                   scoring='accuracy',
                                   cv=group_kfold.split(X, y, groups),
                                   n_jobs=-1)

        grid_search.fit(X, y)
        best_parameters = grid_search.best_params_

        fold_majority_accuracies = []
        fold_accuracies = []  # To store accuracies for each fold
        
        for train_index, test_index in group_kfold.split(X, y, groups):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            test_groups = np.array(groups)[test_index] # Assuming groups is a Pandas Series

            best_model = grid_search.best_estimator_
            best_model.fit(X_train, y_train)
            predictions = best_model.predict(X_test)

            # Calculate majority vote for each group
            majority_votes = []
            for group in np.unique(test_groups):
                group_indices = np.where(test_groups == group)
                group_preds = predictions[group_indices]
                most_common = Counter(group_preds).most_common(1)[0][0]
                majority_votes.extend([most_common] * len(group_indices[0]))

            # Calculate accuracy of the majority vote
            majority_accuracy = accuracy_score(y_test, majority_votes)
            fold_majority_accuracies.append(majority_accuracy)
            
            # Get simple fold accuracy
            fold_accuracy = accuracy_score(y_test, predictions)
            fold_accuracies.append(fold_accuracy)

        best_accuracy = np.mean(fold_majority_accuracies)
        
        result.append({
            'classifier': classifier_name,
            'best_parameters': best_parameters,
            'best_accuracy': best_accuracy,
            'fold_accuracies': fold_majority_accuracies  # Add this line to store accuracies for each fold
        })

    df = pd.DataFrame(result)
    return df


