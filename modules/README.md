# Modules for ML (Working)


  * `robust_evaluate.py`
  
	**A quick way to robustly evaluate signal in a classification task.**
	  
	Given X, y, this script runs a Logistic Regression and evaluates it with
	nested cross-validation.

	Useful for smaller data (less than thousands of records) and potentially
	lots of features, to quickly identify if there is a signal.

    **Usage**


	```python
	from robust_evaluate import Evaluator
	
    est = LogisticRegressionCV()
	
	ev = Evaluator(est, n_repeats=3, n_jobs=2)
	ev.train_evaluate(X, y)
	print(ev.scores_)

	# if you have additional holdout data
	ev.train_predict(X, y, X_holdout, y_holdout)
	print(ev.holdout_scores_)
	```

    **Example with synthetic data**
	
	Usage: `python robust_evaluate.py`, add `--help` for details
	 

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
	modules$ python main.py --holdout --n_samples 2000 --n_repeats 3 --n_jobs 4
	Creating data...
	Evaluating...
	                         mean    std  n_scores
	test_roc_auc            0.926  0.013      15.0
	test_average_precision  0.925  0.014      15.0
	test_neg_log_loss      -0.348  0.032      15.0
	test_accuracy           0.848  0.015      15.0
	test_recall             0.864  0.020      15.0
	test_precision          0.845  0.023      15.0
	test_f1                 0.854  0.014      15.0
	test_f1_weighted        0.848  0.015      15.0
	
	Checking holdout...
	roc_auc_score              0.925549
	average_precision_score    0.918970
	neg_log_loss              -0.349274
	accuracy_score             0.857500
	recall_score               0.847826
	precision_score            0.843243
	f1_score                   0.845528
	f1_score_weighted          0.857528
	```

	
	
