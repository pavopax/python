[Github](https://github.com/pavopax/python)

# Python snippets for modeling

Jump to: [Modeling](#modeling) / [Preproc](#preproc) / [Data](#data) / [Utils](#utils)

# Modeling

Useful utilities for classification:

```python

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
```

# Preproc

### PCA on subset of columns in Pipeline using DataFrameMapper

Simple example:

```python
from sklearn_pandas import DataFrameMapper

bm_cols = [col for col in X.columns if 'fluid' in col]
other_cols = list(set(X.columns) - set(bm_cols))

bm_pca = DataFrameMapper([
    (bm_cols, PCA(n_components=20)),
	(other_cols, None)
])
```

Then, you can fit/transform the DataFrameMapper object, or put it in a pipeline
and fit/transform that:


```python

X = bm_pca.fit_transform(X)

## or:

pipe = Pipeline([
    ('biomarker_pca', bm_pca),
    ('model', LinearRegression())
])

pipe.fit(X_train, y_train)
```

Real use case:

```python
featurizer = DataFrameMapper([
    (bm_cols,    make_pipeline(StandardScaler(), PCA(n_components=10))),
    (other_cols, StandardScaler()))
])

pipe = Pipeline([
    ('featurizer', featurizer),
    ('model', CoxPHSurvivalAnalysis())
])

pipe.fit(X_train, y_train)
```

Docs: [github.com/scikit-learn-contrib/sklearn-pandas#usage](https://github.com/scikit-learn-contrib/sklearn-pandas#usage)

### Remove features with low variance

Automatically. Use `VarianceThreshold()`


First, do it stand-alone, if you want to know which X's are kept (removed)

```python
vt = VarianceThreshold(threshold=0.1)

# remove some columns from X
# Xv is now a numpy array, and has no column names...
Xt = vt.fit_transform(X)

# ... so use get_support() to show kept columns
X.columns[vt.get_support()]
```

Best practice: apply a preprocessing pipeline separately from a modeling pipeline. 

Rationale

  * In pre-processing, you may lose column names at converstion from numpy
    array to pandas data frame
  * keep this process separate so you can use something like `get_support()` to
    get back column names downstream
	
Disadvantage

  * if you need to tune a pre-processing pipeline (eg, thresholds in `PCA` or
    `VarianceThreshold()`), then you may need to do it as one pipe. see below


Example pre-processing pipelines, with final modeling pipeline

```python

sl = make_sl(config, dfs, query=query)
X0, y = make_X_y_from_sl(config, sl, outcome=outcome)
# pipe 1
preproc = make_pipeline(Imputer(), StandardScaler())
X1 = preproc.fit_transform(X0)
# pipe 2
vt = VarianceThreshold(0)
X = vt.fit_transform(X1)
# here, you have column names
vt_cols = X0.columns[vt.get_support()]

X, y, vt_cols = get_X_y(config, dfs=dfs, query=query, outcome=outcome)

X_train, X_test, y_train, y_test = train_test_split(X, y,
    test_size=0.33, random_state=seed)

# pipe 3 - modeling
pipe_gbs = GradientBoostingSurvivalAnalysis()

params_gbs = dict(
	n_estimators=GBS_TREES,
	max_depth=GBS_DEPTH,
	max_features=GBS_MAXF
)

gbs_gcv = GridSearchCV(pipe_gbs, params_gbs, cv=GCV_CV, n_jobs=JOBS)

gbs_gcv.fit(X_train, y_train)


```



Alternatively, you can have one pipeline

```python
THRESHOLD = 0.1

pipe = make_pipeline(
    VarianceThreshold(THRESHOLD),
	LinearRegression()
	)

pipe.fit(X_train, y_train)
```

### Impute 

```python
# Replace NaN values by mean values
from sklearn.preprocessing import Imputer

df.fillna(value=nan, inplace=True)
imputer = Imputer()
transformed_X = imputer.fit_transform(X)
```

# Scalers and outliers

See `scalers-outliers.md` in this folder on GitHub



# Data

### Select categorical columns

```python
text_columns = X.dtypes.pipe(lambda x: x[x == 'object']).index
```

### Filtering in Pandas

*You have a data frame "df"*

### Filter on column

One way:

```python
df.query('trt=="Atezo"')
```

But, to do this programatically (eg, for loop) they above doesn't seem to work due to the quotes. Therefore, try:

```python
df[df.trt==trt]

df[df.trt.isin([3, 6])]
```
EG:
```python
dfs = {}

for trt in trts:
   dfs[trt] = df[df.trt==trt]
```

### Filter on index

If you have and are using an index (one record per value), you can try:

```python
df.filter(like="train", axis=1)
```

*This will give error if the output's index (one-level index, not multilevel) is not one record per index
value. Usually, a (single-level) index is an ID variable that is unique to each row. If this
is not true, try `df.reset_index()` and then filter on the column (instead of index) as shown above.*

*If you ARE using a multilevel index, you would need to specify the level to filter on. For this, following rule one in `import this`, I had: `dfs[trt] =  X[X.index.get_level_values('trt').isin([trt])]`.*


### filter using logical

source: https://stackoverflow.com/a/42113965/3217870

```python
def response_to_0_1(row):
    if row['response'] == "PR":
        return(1)
    elif row['response'] == "CR":
        return(1)
    else:
        return(0)

outcomes = outcomes.assign(resp_0_1=outcomes.apply(response_to_0_1, axis=1))
```






### Select


Select difference of column names:

```python
X_features_trt_df.columns.difference(['TRT01P'])

```

### Collapse nested lists inside data frame

`res` data frame has column "features", where each cell has a list. I want to
collect all items inside all those lists, and then count them.

```python
genes = []

for i in range(len(res)):
    genes.append(res.features[i])

genes = [g for x in genes for g in x]

res = pd.Series(genes).value_counts()

# view results
res.head()
res.plot()
```


# Utils

### load custom modules 

Possibly from a relative path

I am at `.` and have file `../code/helpers.py`

(`%magic` is for Jupyter Notebook only)

```python
# https://ipython.org/ipython-doc/3/config/extensions/autoreload.html
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('../code')

from helpers import write_result_record, read_results
```

### Append records to file with open(file, mode='a')

Instructions to read it back are also below.

You plan on running lots of models. Here's a quick way to save the results.

I run different `pipe`'s in a Jupyter cell, then run this to save a record.


```python
record = {
    'test' : pipe.score(X_test, y_test),
    'train' : pipe.score(X_train, y_train),
    'rows' : X.shape[0],
    'cols' : X.shape[1],
    'model': str(pipe.steps[-1]),
    'full_pipe' : str(pipe.steps).replace("\n", "")
}


# use mode='a' to append records!
# don't forget to add newline

with open('results.json', mode='a') as f:
    f.write(json.dumps(record) + "\n")

# read it back as DF with lines=True (pandas 0.19)
pd.read_json('results.json', lines=True)

```


### Shuffle an array

Shuffle an array. You **don't** `assign` the shuffle to a new vector!

```python
y_scramble = copy.copy(y)
y_scramble.dead = y.dead.astype('int')
np.random.shuffle(y.dead)
y.dead.astype('bool')
```

### Nulls

```python
df.isnull().sum().sum()

df.isnull().sum()
```



### arg parse

https://docs.python.org/3/library/argparse.html

```python
# your arguments:
args = parser.parse_args()

# print:
print(args)

# print as dict
print(vars(args))

# print with _ delimiter (useful to add to filename)
print("_".join(vars(args).values()))

```


### Logging

inside python script:

```python
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
```

Inside notebooks, see
[https://stackoverflow.com/a/18809051/3217870](https://stackoverflow.com/a/18809051/3217870)


### throw error in function

```python
if coxnet_alpha is not None:
    if type(coxnet_alpha) is list:
		## RAISE ERROR and stop function
        raise TypeError("'coxnet_alpha' needs to be float, not list")
    res = res[coxnet_alpha]
    if only_non_zero:
        return(res[res != 0])
    else:
        return(res)

```


# References/Inspirations

[chrisalbon.com/#python](https://chrisalbon.com/#python)

[jeffdelaney.me/blog/useful-snippets-in-pandas/](https://jeffdelaney.me/blog/useful-snippets-in-pandas/)
