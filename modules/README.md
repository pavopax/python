# Modules for ML (Working)


  * `robust_evaluate.py`
  
	**A quick way to robustly evaluate signal in a classification task.**
	  
	Given X, y, this script runs a Logistic Regression and evaluates it with
	nested cross-validation.

	Useful for smaller data (less than thousands of records) and potentially
	lots of features, to quickly identify if there is a signal.

    Usage:

    Load the module into your environment to apply on your own data, or run on
    command line with synthetic data:

    `python robust_evaluate.py` 

	add `--help` for details
	 
	Example:

    ```bash
	modules$ python robust_evaluate.py --imbalanced
	Creating data...
	Evaluating...
	                         mean    std  n_scores
	test_roc_auc            0.963  0.036       5.0
	test_average_precision  0.800  0.170       5.0
	test_neg_log_loss      -0.091  0.028       5.0
	test_accuracy           0.965  0.012       5.0
	test_recall             0.300  0.245       5.0
	test_precision          0.600  0.490       5.0
	test_f1                 0.400  0.327       5.0
	test_f1_weighted        0.953  0.022       5.0
	```


  * `main.py`

	**More experiments using `robust_evaluate.py`**
  
	Example

	```bash
	modules$ python main.py --holdout --n_samples 500 --n_repeats 3 --n_jobs 4
	Creating data...
	Evaluating...
	                         mean    std  n_scores
	test_roc_auc            0.892  0.035      15.0
	test_average_precision  0.866  0.051      15.0
	test_neg_log_loss      -0.425  0.075      15.0
	test_accuracy           0.827  0.036      15.0
	test_recall             0.840  0.059      15.0
	test_precision          0.817  0.043      15.0
	test_f1                 0.827  0.038      15.0
	test_f1_weighted        0.826  0.036      15.0
	
	Checking holdout...
	roc_auc_score              0.940000
	average_precision_score    0.935374
	neg_log_loss              -0.322117
	accuracy_score             0.860000
	recall_score               0.820000
	precision_score            0.891304
	f1_score                   0.854167
	f1_score_weighted          0.859776
	```

	
	
