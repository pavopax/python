import argparse
import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegressionCV


class Evalutor:

    """Evaluate X, y, returning scores in self.scores_"""

    def __init__(self, n_jobs, add_pca):
        self.n_jobs = n_jobs
        self.add_pca = add_pca

    def summarise_scores(self, scores: dict, round_places: int = 3,
                         return_df: bool = True) -> pd.DataFrame:
        """scores is a result from cross_validate"""
        metric_names = [x for x in scores.keys() if "test_" in x]
        summary = dict()
        for metric in metric_names:
            scores_for_this_metric = scores[metric]
            summary[metric] = {
                'mean': scores_for_this_metric.mean().round(round_places),
                'std': scores_for_this_metric.std().round(round_places),
                'n_scores': len(scores_for_this_metric)
            }
        if return_df:
            return(pd.DataFrame(summary).T)
        else:
            return(summary)

    def train_evaluate(self, X, y, return_summary=True):
        print("Processing...")

        # TODO convert into arguments
        n_components = 0.95
        seed = 1
        outer_cv = 3
        inner_cv = 5
        n_repeats = 2
        max_iter = 300

        score_metrics = ["average_precision", "roc_auc", "recall", "precision",
                         "accuracy", "f1", "f1_weighted", "neg_log_loss"]

        inner_scoring = "roc_auc"

        lrcv = LogisticRegressionCV(cv=inner_cv, scoring=inner_scoring,
                                    n_jobs=n_jobs, class_weight="balanced",
                                    random_state=seed, max_iter=max_iter)

        if add_pca == "yes":
            estimator_pipe = make_pipeline(
                StandardScaler(),
                PCA(n_components=n_components, random_state=seed),
                lrcv
            )
        elif add_pca == "no":
            estimator_pipe = make_pipeline(
                StandardScaler(),
                lrcv
            )

        rkf = RepeatedStratifiedKFold(n_splits=outer_cv, n_repeats=n_repeats,
                                      random_state=seed)

        print("Evaluating...")
        scores_dict = cross_validate(estimator_pipe, X, y, cv=rkf,
                                     scoring=score_metrics, n_jobs=n_jobs)

        if return_summary:
            self.scores_ = self.summarise_scores(scores_dict, return_df=True)
        else:
            self.scores_ = scores_dict

        return(self)


if __name__ == '__main__':
    from sklearn.datasets import make_classification

    parser = argparse.ArgumentParser(description="Arguments")
    parser.add_argument('--n_jobs', default=2,
                        help="""Number of cores to use. Default is 2.""")
    parser.add_argument('--add_pca', default="no",
                        help="""Add PCA? yes or no""")
    parser.add_argument('--imbalanced', default="no",
                        help="""Create imbalanced data? yes or no""")

    args = parser.parse_args()

    n_jobs = np.int(args.n_jobs)
    add_pca = str(args.add_pca)
    imbalanced = str(args.imbalanced)

    print("Creating data...")
    weights = None
    if imbalanced == "yes":
        weights = [1.9,  0.8]

    X, y = make_classification(n_samples=200, n_features=20, n_informative=15,
                               n_classes=2, weights=weights, random_state=1)

    evaluator = Evalutor(n_jobs=n_jobs, add_pca=add_pca)
    evaluator.train_evaluate(X, y, return_summary=True)
    print(evaluator.scores_)
