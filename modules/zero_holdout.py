import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingClassifier
#
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier


from robust_evaluate import Evaluator

imbalanced = False
holdout_size = 0.3
random_state = 99

weights = None
if imbalanced:
    weights = [1.9,  0.8]

X_, y_ = make_classification(n_samples=100, n_features=20, n_informative=2,
                             n_classes=2, weights=weights,
                             random_state=random_state)

X, X_holdout, y, y_holdout = train_test_split(
    X_, y_, test_size=holdout_size, random_state=random_state)


# which one if imbalanced=True !?
class_weight = "balanced"
class_weight = None

est = make_pipeline(
    StandardScaler(),
    LogisticRegressionCV(cv=5, scoring="accuracy",
                         n_jobs=1,
                         class_weight=class_weight,
                         random_state=99, max_iter=300)

)


ev = Evaluator(est, n_repeats=3, n_jobs=2)
ev.train_evaluate(X, y)
print(pd.DataFrame(ev.scores_).T)

ev.train_predict(X, y, X_holdout, y_holdout)
print(pd.Series(ev.holdout_scores_))


# ============================================================================
# gbm
# ============================================================================
est = make_pipeline(
    GradientBoostingClassifier()

)


ev = Evaluator(est, n_repeats=3, n_jobs=2)
ev.train_evaluate(X, y)
print(pd.DataFrame(ev.scores_).T)

ev.train_predict(X, y, X_holdout, y_holdout)
print(pd.Series(ev.holdout_scores_))


# ============================================================================
# hgbm
# ============================================================================
est = make_pipeline(
    HistGradientBoostingClassifier()

)


ev = Evaluator(est, n_repeats=3, n_jobs=2)
ev.train_evaluate(X, y)
print(pd.DataFrame(ev.scores_).T)

ev.train_predict(X, y, X_holdout, y_holdout)
print(pd.Series(ev.holdout_scores_))


for i in range(20):
    print(i)
    print(np.corrcoef(X_[:, i], y_))
