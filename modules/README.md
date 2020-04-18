# Modules for ML (Working)


  * `robust_evaluate.py`
  
     **A quick way to robustly evaluate signal in a classification task.**
	 
     Given X, y, this script runs a Logistic Regression and evaluates it with
     nested, repeated cross-validation.

	 Useful for smaller data (less than thousands of records) and potentially
     lots of features, to quickly identify if there is a signal.

     Usage:

     Load the module into your environment to apply on your own data, or run on
     command line with synthetic data:

     `python robust_evaluate.py` 

	 add `--help` for more
	 
	 Example:

     ```
	 modules$ python robust_evaluate.py --imbalanced 
	 Creating data...
	 Evaluating...
	                          mean    std  n_scores
	 test_average_precision  0.547  0.153       6.0
	 test_roc_auc            0.762  0.114       6.0
	 test_recall             0.542  0.172       6.0
	 test_precision          0.358  0.294       6.0
	 test_accuracy           0.877  0.049       6.0
	 test_f1                 0.375  0.152       6.0
	 test_f1_weighted        0.898  0.035       6.0
	 test_neg_log_loss      -0.619  0.116       6.0
	 ```

