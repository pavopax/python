from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (GridSearchCV, cross_validate,
                                     train_test_split)
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def main():
    print("Loading...")
    random_state = 1
    verbose = 0

    n_jobs = 4
    inner_cv = 5
    outer_cv = 3
    inner_metric = 'roc_auc'
    outer_metrics = ['roc_auc', 'average_precision', 'f1']
    
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5,
                               n_redundant=0, random_state=random_state)

    param_grid = dict(
        randomforestclassifier__n_estimators=[100, 200],
        randomforestclassifier__min_samples_leaf=[1, 2]
    )

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(random_state=random_state)
    )

    estimator = GridSearchCV(estimator=model, param_grid=param_grid,
                             scoring=inner_metric, cv=inner_cv, refit=True,
                             return_train_score=False, verbose=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1/outer_cv, random_state=random_state)

    print("Training...")
    estimator.fit(X_train, y_train)

    cv_results = cross_validate(estimator, X, y, scoring=outer_metrics,
                                cv=outer_cv, verbose=verbose,
                                return_train_score=False,
                                n_jobs=n_jobs,                            
                                return_estimator=True)
    
    print("Evaluating...")
    preds_probs = estimator.predict_proba(X_test)[:,1] # take probs for y==1

    importances = [x.best_estimator_.named_steps['randomforestclassifier'].feature_importances_
                   for x in cv_results['estimator']]

    print("X dimensions:")
    print(X.shape)
    print("ROC AUC, from single train/test split:")
    print(roc_auc_score(y_test, preds_probs))
    print("ROC AUC, from nested CV (mean, std, count):")
    print(cv_results['test_roc_auc'].mean(), cv_results['test_roc_auc'].std(), len(cv_results['test_roc_auc']))

    print("Feature Importances sum to 1")
    print([x.sum() for x in importances])

    # deploy
    # estimator.fit(X, y);
    # X_new = get_prod_data()
    # estimator.predict(X_new)


if __name__ == "__main__":
    main()
