from dataclasses import dataclass, field
from typing import List

import argparse

import sklearn as sk
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import (average_precision_score,
                             roc_auc_score, recall_score,
                             precision_score, accuracy_score,
                             f1_score, accuracy_score, log_loss)


# https://realpython.com/python-data-classes/
@dataclass
class Evaluator:

    """Evaluate X, y, returning scores in self.scores_

    Optionally, you can predict (and evaluate) on an additional holdout dataset

    """
    estimator: sk.base.BaseEstimator
    # setting a default list with a dataclass is not so intuitive
    score_metrics: List[str] = field(
        default_factory=lambda: ["roc_auc",
                                 "average_precision",
                                 "neg_log_loss",
                                 "accuracy", "recall",
                                 "precision", "f1",
                                 "f1_weighted"])
    outer_cv: int = 5
    n_repeats: int = 1
    n_jobs: int = 1
    random_state: int = 1

    def train_evaluate(self, X, y, summarise_scores=True):
        # (are these assertions pythonic?)
        assert type(
            self.score_metrics) == list, "score_metrics must be a List[str]"
        assert all([x in sk.metrics.SCORERS.keys() for
                    x in self.score_metrics]), "all score metrics must match sk.metrics.SCORERS.keys()"

        rkf = RepeatedStratifiedKFold(n_splits=self.outer_cv,
                                      n_repeats=self.n_repeats,
                                      random_state=self.random_state)
        scores_dict = cross_validate(self.estimator, X, y, cv=rkf,
                                     scoring=self.score_metrics,
                                     n_jobs=self.n_jobs)

        if summarise_scores:
            self.scores_ = self.summarise_scores(scores_dict)
        else:
            self.scores_ = scores_dict

    def train_predict(self, X, y, X_new, y_new=None):
        self.estimator_ = self.estimator.fit(X, y)
        # uses estimator_ not estimator
        self.preds_ = self.estimator_.predict(X_new)
        self.preds_prob_ = self.estimator_.predict_proba(X_new)[:, 1]

        if y_new is not None:
            self.holdout_scores_ = self._compute_holdout_scores(y_new)

    def summarise_scores(self, scores: dict, round_places: int = 3) -> dict:
        """scores is a result from cross_validate()"""
        metric_names = [x for x in scores.keys() if "test_" in x]
        summary = dict()
        for metric in metric_names:
            scores_for_this_metric = scores[metric]
            summary[metric] = {
                'mean': scores_for_this_metric.mean().round(round_places),
                'std': scores_for_this_metric.std().round(round_places),
                'n_scores': len(scores_for_this_metric)
            }
        return(summary)

    def _compute_holdout_scores(self, y_new) -> dict:
        return dict(
            # use preds_prob_
            roc_auc_score=roc_auc_score(y_new, self.preds_prob_),
            average_precision_score=average_precision_score(
                y_new, self.preds_prob_),
            neg_log_loss=-log_loss(y_new, self.preds_prob_),
            # use preds_
            accuracy_score=accuracy_score(y_new, self.preds_),
            recall_score=recall_score(y_new, self.preds_),
            precision_score=precision_score(y_new, self.preds_),
            f1_score=f1_score(y_new, self.preds_),
            f1_score_weighted=f1_score(
                y_new, self.preds_, average="weighted")
        )


if __name__ == '__main__':
    import pandas as pd

    from sklearn.datasets import make_classification
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa
    from sklearn.ensemble import HistGradientBoostingClassifier

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--imbalanced', action="store_true",
                        help="""Create imbalanced data?""")
    parser.add_argument('--holdout', action="store_true",
                        help="""Withold some data for a holdout evaluation?""")
    parser.add_argument('--holdout_only', action="store_true",
                        help="""Only report holdout scores and do not show nested CV scores?""")
    parser.add_argument('--clf', default='lrcv', nargs="+", choices=['lrcv', 'gbm', 'hgbm'],
                        help="""Classifier: any of lrcv, gbm, hgbm""")
    parser.add_argument('--n_samples', default=100,
                        help="""Number of samples in data (includes --holdout). Default is 300.""")
    parser.add_argument('--n_features', default=20,
                        help="""Number of features in data. Default is 20.""")
    parser.add_argument('--n_informative', default=16,
                        help="""Number of informative features in data. Default is 16.""")
    parser.add_argument('--outer_cv', default=5,
                        help="""K for outer KFold CV nested CV. Default is 5.""")
    parser.add_argument('--n_repeats', default=1,
                        help="""Number of repeats for nested CV. Default is 1.""")
    parser.add_argument('--add_pca', action="store_true",
                        help="""Add PCA?""")
    parser.add_argument('--n_jobs', default=2,
                        help="""Number of cores to use. Default is 2.""")
    parser.add_argument('--random_state', default=99,
                        help="""Seed for sklearn""")

    args = parser.parse_args()

    imbalanced = args.imbalanced
    holdout = args.holdout
    holdout_only = args.holdout_only
    clf = args.clf
    n_samples = int(args.n_samples)
    n_features = int(args.n_features)
    n_informative = int(args.n_informative)
    outer_cv = int(args.outer_cv)
    n_repeats = int(args.n_repeats)
    add_pca = args.add_pca
    n_jobs = int(args.n_jobs)
    random_state = int(args.random_state)

    print("Creating data...")
    weights = None
    if imbalanced:
        weights = [1.9,  0.8]

    X_, y_ = make_classification(n_samples=n_samples, n_features=20,
                                 n_informative=18, n_classes=2,
                                 weights=weights, random_state=random_state)

    if holdout:
        holdout_size = 0.2
        X, X_holdout, y, y_holdout = train_test_split(
            X_, y_, test_size=holdout_size, random_state=random_state)
    else:
        X = X_
        y = y_

    # TODO convert into arguments
    n_components = 0.95
    inner_cv = 5
    max_iter = 300
    class_weight = None

    lrcv = LogisticRegressionCV(cv=inner_cv, scoring="accuracy",
                                n_jobs=1,
                                class_weight=class_weight,
                                random_state=random_state, max_iter=max_iter)
    gbm = GradientBoostingClassifier(n_estimators=100)
    hgbm = HistGradientBoostingClassifier(max_iter=100)

    print("Samples: ", str(n_samples))
    print("Features: ", str(n_features))
    print("Informative: ", str(n_informative))

    # convert single clf argument to list, if only one was passed
    if type(clf) is not list:
        clf = [clf]

    for clf_ in clf:
        print("Evaluating classifier: ", clf_)

        if add_pca:
            pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=random_state),
                eval(clf_)
            )
        else:
            pipe = make_pipeline(
                StandardScaler(),
                eval(clf_)
            )

        ev = Evaluator(estimator=pipe, outer_cv=outer_cv, n_repeats=n_repeats,
                       n_jobs=n_jobs, random_state=random_state)
        ev.train_evaluate(X, y, summarise_scores=True)

        if not holdout_only:
            print(pd.DataFrame(ev.scores_).T)

        if holdout:
            print("\nChecking holdout...")
            ev.train_predict(X, y, X_holdout, y_holdout)
            print(pd.Series(ev.holdout_scores_))
        print("\n")
