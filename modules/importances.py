"""Extract feature importances or coefs from cross_validate(.., return_estimators=True)"""
import pandas as pd


def get_importances_from_cv(cv_results, coef_flatten=True, column_names=None,
                            debug=False):

    models = cv_results['estimator']

    if debug:
        print(f"Found {len(models)} models in cv_results")

    # coef_ or feature_imporances_

    # TODO: [-1] is needed for pipeline. It seems to be ok without pipeline
    # too, right?
    if 'coef_' in dir(models[0]):
        if coef_flatten:
            importances = [x.coef_.flatten() for x in models]
        else:
            # TODO: flatten() may not be appropriate with regularized results
            # (coefs_ for each regularization parameter)
            importances = [x.coef_ for x in models]
    else:
        importances = [x.feature_importances_ for x in models]

    # remember, there are MULTIPLE models, and so multiple sets of importances
    # concatenate them on top of each other into single df
    importances_dfs = [pd.DataFrame(
        dict(feature=column_names, importance=x)) for x in importances]
    importances_df = pd.concat(importances_dfs)
    return(importances_df)


def summarise_importances(df, summary_stats: list = ['count', 'mean', 'std'],
                          sort=True):
    assert 'mean' in summary_stats, (
        "summary_stats must include at least <mean>")
    results = df.groupby(
        'feature')['importance'].describe()[summary_stats]
    results['abs_mean'] = results['mean'].abs()
    results = results[['count', 'mean', 'abs_mean', 'std']]
    if sort:
        results = results.sort_values('abs_mean', ascending=False)
    return(results)
