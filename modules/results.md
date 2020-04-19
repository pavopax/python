# Results

# 1

```
modules$ python main.py --hold --clf lrcv gbm hgbm --n_samples 100 --n_repeats 3 --n_jobs 3
Creating data...
Samples:  100
Features:  20
Informative:  16
Evaluating classifier:  lrcv
                         mean    std  n_scores
test_roc_auc            0.785  0.127      15.0
test_average_precision  0.798  0.141      15.0
test_neg_log_loss      -1.218  1.409      15.0
test_accuracy           0.707  0.112      15.0
test_recall             0.705  0.205      15.0
test_precision          0.679  0.114      15.0
test_f1                 0.683  0.154      15.0
test_f1_weighted        0.702  0.117      15.0

Checking holdout...
roc_auc_score              0.989011
average_precision_score    0.994505
neg_log_loss              -0.304300
accuracy_score             0.950000
recall_score               0.923077
precision_score            1.000000
f1_score                   0.960000
f1_score_weighted          0.950667
dtype: float64


Evaluating classifier:  gbm
                         mean    std  n_scores
test_roc_auc            0.830  0.101      15.0
test_average_precision  0.840  0.103      15.0
test_neg_log_loss      -0.676  0.281      15.0
test_accuracy           0.794  0.091      15.0
test_recall             0.794  0.147      15.0
test_precision          0.788  0.123      15.0
test_f1                 0.782  0.107      15.0
test_f1_weighted        0.792  0.093      15.0

Checking holdout...
roc_auc_score              0.868132
average_precision_score    0.941497
neg_log_loss              -0.550303
accuracy_score             0.800000
recall_score               0.923077
precision_score            0.800000
f1_score                   0.857143
f1_score_weighted          0.790476
dtype: float64


Evaluating classifier:  hgbm
                         mean    std  n_scores
test_roc_auc            0.692  0.114      15.0
test_average_precision  0.712  0.103      15.0
test_neg_log_loss      -0.726  0.182      15.0
test_accuracy           0.679  0.087      15.0
test_recall             0.667  0.153      15.0
test_precision          0.666  0.090      15.0
test_f1                 0.658  0.103      15.0
test_f1_weighted        0.676  0.088      15.0

Checking holdout...
roc_auc_score              0.835165
average_precision_score    0.904092
neg_log_loss              -0.532478
accuracy_score             0.800000
recall_score               0.769231
precision_score            0.909091
f1_score                   0.833333
f1_score_weighted          0.804167
dtype: float64
```

# 2

```
modules$ python main.py --hold --clf lrcv gbm hgbm --n_samples 1000 --n_repeats 3 --n_jobs 3
Creating data...
Samples:  1000
Features:  20
Informative:  16
Evaluating classifier:  lrcv
                         mean    std  n_scores
test_roc_auc            0.862  0.029      15.0
test_average_precision  0.851  0.046      15.0
test_neg_log_loss      -0.467  0.047      15.0
test_accuracy           0.794  0.028      15.0
test_recall             0.806  0.050      15.0
test_precision          0.794  0.028      15.0
test_f1                 0.799  0.031      15.0
test_f1_weighted        0.793  0.028      15.0

Checking holdout...
roc_auc_score              0.887784
average_precision_score    0.849480
neg_log_loss              -0.427530
accuracy_score             0.790000
recall_score               0.795455
precision_score            0.744681
f1_score                   0.769231
f1_score_weighted          0.790572
dtype: float64


Evaluating classifier:  gbm
                         mean    std  n_scores
test_roc_auc            0.931  0.015      15.0
test_average_precision  0.923  0.026      15.0
test_neg_log_loss      -0.345  0.027      15.0
test_accuracy           0.858  0.017      15.0
test_recall             0.879  0.037      15.0
test_precision          0.850  0.026      15.0
test_f1                 0.863  0.017      15.0
test_f1_weighted        0.858  0.017      15.0

Checking holdout...
roc_auc_score              0.936282
average_precision_score    0.925400
neg_log_loss              -0.346646
accuracy_score             0.830000
recall_score               0.875000
precision_score            0.770000
f1_score                   0.819149
f1_score_weighted          0.830614
dtype: float64


Evaluating classifier:  hgbm
                         mean    std  n_scores
test_roc_auc            0.954  0.014      15.0
test_average_precision  0.952  0.018      15.0
test_neg_log_loss      -0.281  0.052      15.0
test_accuracy           0.891  0.018      15.0
test_recall             0.909  0.028      15.0
test_precision          0.882  0.023      15.0
test_f1                 0.895  0.018      15.0
test_f1_weighted        0.891  0.018      15.0

Checking holdout...
roc_auc_score              0.963170
average_precision_score    0.947195
neg_log_loss              -0.275533
accuracy_score             0.885000
recall_score               0.943182
precision_score            0.821782
f1_score                   0.878307
f1_score_weighted          0.885412
dtype: float64
```
