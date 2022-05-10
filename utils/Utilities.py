import copy
import math
import multiprocessing
import sys
from datetime import datetime
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.combine import SMOTETomek
from joblib import Parallel, delayed
from matplotlib import pyplot as plt

from utils.multiclass import multiclass_metrics as mm

np.set_printoptions(threshold=sys.maxsize)


def plot_data_set(counts, title, colors=['#8366ac', '#ca5959', '#e4be6c', '#5960ae', '#75aa68'], xlabel='Rating', ylabel='Workouts', type=np.nan):
    a = [i for (i, j) in counts]
    b = [j for (i, j) in counts]

    plt.figure(figsize=(12, 6), dpi=200)
    if type == 'line':
        plt.plot(a, b)
        plt.xticks(range(min(a), math.ceil(max(a)) + 1), fontsize=22)
    else:
        plt.bar(a, b, color=colors, width=.4)
        plt.xticks(fontsize=22)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)

    plt.yticks(fontsize=22)

    plt.title(title)
    plt.grid(True)
    plt.show()


def get_rid_of_nulls(value):
    if isinstance(value, str):
        return 0
    if value != value:
        return 0
    else:
        return value


def parseDate(raw_string_date):
    POSSIBLE_FORMATS = ['%m/%d/%Y', '%Y/%m/%d']  # all the formats the date might be in

    for format in POSSIBLE_FORMATS:
        try:
            parsed_date = datetime.strptime(raw_string_date, format).year  # try to get the date

            return parsed_date  # if correct format, don't test any other formats
        except ValueError:

            pass  # if incorrect format,
    return np.nan  # datetime.strptime('01/01/2019', '%m/%d/%Y').year


def calculatePaceFromSpeed(speeddone):
    if speeddone > 0:
        avgPaceSeconds = 3600 / speeddone;
        return avgPaceSeconds
    elif speeddone == 0:
        return 0
    return np.nan


# ------------------------------------------------------------------------------
# accept a dataframe, remove outliers, return cleaned data in a new dataframe
# see http://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm
# ------------------------------------------------------------------------------
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out


def box_plot_column(pdf, column, target, value):
    df = pdf[[column]][pdf[target] == value]
    df = remove_outlier(df, column)
    df.boxplot(figsize=(24, 16))


# Python code to count the number of occurrences 
def countX(lst, x):
    count = 0
    for ele in lst:
        if ele == x:
            count = count + 1
    return count


def get_activity_percent(workout, activity_label):
    activities = workout.w_a_label.values
    return countX(activities, activity_label) / len(activities)


def calculateBMI(weight, height):
    if weight == 0 or height == 0:
        return np.nan
    return weight / (height / 100) ** 2


def fill_null_mean(df):
    columns = df.columns
    values = {}
    for c in columns:
        if c in df.select_dtypes(include=['number']).columns:
            values[c] = np.mean(df[c].dropna().values)

    df.fillna(value=values, inplace=True)
    try:
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].astype(np.float64)
        df = df[df.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    except Exception as e:
        print(e)
        pass
    return df.dropna(axis=1, how='all')


def calculateGap(pianificato, fatto):
    if pianificato == 0:
        return 0
    return (pianificato - fatto) / pianificato


def calculateGapBetweenColumns(ddf, col1, col2):
    gap = []
    col1Values = ddf[col1].values
    col2Values = ddf[col2].values
    for x, y in zip(col1Values, col2Values):
        gap.append(calculateGap(x, y))
    return gap


def calculateFeatureWellDone(pianificato, fatto):
    if pianificato == 0:
        return 0
    if pianificato != 0 and pianificato - fatto == 0:
        return 1
    return -1


def calculateActivityWellDone(row):
    if row.w_a_type == 'TIME':
        return calculateFeatureWellDone(row.w_a_time, row.r_timedone)
    if row.w_a_type == 'DISTANCE':
        return calculateFeatureWellDone(row.w_a_distance, row.r_distancedone)
    if row.w_a_type == 'DISTANCE_TIME':
        return calculateFeatureWellDone(row.w_a_pace, row.r_pace)
    return 0


def calculateWorkoutWelldone(workout):
    welldone = len(workout[workout.a_welldone == 1])
    n_activities = len(workout[workout.a_welldone != 0])
    if n_activities != 0:
        return welldone / n_activities
    return 0


def activityHasObjectif(row):
    if row.w_a_type != 'EXTRA':
        return 1
    return 0


def calculatePhasObjectif(workout):
    has_objectif = len(workout[workout.has_objectif == 1])
    n_activities = len(workout)
    if n_activities != 0:
        return has_objectif / n_activities
    return 0


def get_weight_situation(bmi):
    if bmi >= 40:
        return 8
    if 40 > bmi >= 35:
        return 7
    if 35 > bmi >= 30:
        return 6
    if 30 > bmi >= 25:
        return 5
    if 25 > bmi >= 18.5:
        return 4
    if 18.5 > bmi >= 17:
        return 3
    if 17 > bmi >= 16:
        return 2
    if bmi < 16:
        return 1


# Create a function called "chunks" with two arguments, l and n:
def chunkIt(l, n):
    # if the list cannot be devided into n equal chunks remove 
    # the last elements because they are the less relevant in the case of ranking
    
    if len(l)%n is not 0:
        l = l[:len(l)-(len(l)%n)]
    

    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]
        
        
def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r[:k], reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def mean_ndcg_at_k(r,k):
    rs = chunkIt(r,k)
    return np.mean([ndcg_at_k(r,k) for r in rs])


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = list(rs)
    rs = list(np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.sum(out) / len(r)


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def plot_correlation(df):
    # Compute the correlation matrix
    # exclude 'Open' variable
    corr_all = df.corr()

    # Generate a mask for the upper triangle
    # zeros_like returns an array of zeros with the same shape and type as a given array
    mask = np.zeros_like(corr_all, dtype=np.bool)
    # triu_indices_from returns the indices for the upper-triangle of the array it receives as input
    mask[np.triu_indices_from(mask)] = True  # we set as True only the upper triangle

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr_all, mask=mask,
                square=True, linewidths=.5, ax=ax, cmap="BuPu")
    plt.show()


def parallelize(function, inputs):
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(function)(i) for i in inputs)
    return results


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def remove_features_one_by_one(model, features, X_train, y_train, target_col="rating"):
    start_time = time()
    i = len(features)
    models = []
    features_ = []
    print(' \n'+ model.__class__.__name__ )
    print(' - Removing Features One By One and Training ')
    while i >= 0:

        b = ((1 + len(features) - i) * '=' + '> ' + str(int(100 * (len(features) - i) / len(features))) + '%')
        sys.stdout.write('\r' + b)
        fc = copy.deepcopy(features)
        if i < len(features):
            del fc[i]

        trained_model = train_model_with_features(model, fc, X_train, y_train,target_col)
        models.append(copy.deepcopy(trained_model))
        features_.append(copy.deepcopy(fc))
        i = i - 1
    sys.stdout.write('\r' + model.__class__.__name__ + ' - Elapsed time: ' + convertMillis(round((time() - start_time) * 1000)))

    return features_, models


def train_model_with_features(model, features, X, y, target_col="rating"):
    X_train = X[features]
    y_train = y[target_col].values.ravel()
    smt = SMOTETomek(random_state=0)
    X_train, y_train = smt.fit_sample(X_train, y_train)
    if hasattr(model, 'fit'):
        model.fit(X_train, y_train)
        return model
    else:
        raise ValueError("Model has no method fit()")


def print_results(models, features, X_test, y_test, path):
    from sklearn.metrics import accuracy_score
    results = [['Algorithm', 'Accuracy', 'F1-Score', 'F2-Score', 'Recall', 'Precision', 'Informedness', 'RMSE', 'removed_item']]
    print('\n - Printing Results')
    for i in range(len(models)):
        b = ((i + 1) * '=' + '> ' + str(int(100 * (i + 1) / len(models))) + '%')
        sys.stdout.write('\r' + b)
        metrics = mm.MulticlassMetrics(models[i], X_test[features[i]], y_test)
        # metrics.to_string()
        # print('ndcg_at_5     ', ndcg_at_k(models[i].predict(X_test[features[i]]), 5, 0))
        # print('ndcg_at_10    ', ndcg_at_k(models[i].predict(X_test[features[i]]), 10, 0))
        results.append(
                [models[i].__class__.__name__,
                 str(round(accuracy_score(y_test, models[i].predict(X_test[features[i]])),3)),
                 str(round(metrics.f1_score(),3)), 
                 str(round(metrics.f2_score(),3)),
                 str(round(metrics.recall_score(),3)), 
                 str(round(metrics.precision_score(),3)),
                 str(round(metrics.informedness_score(),3)), 
                 str(round(metrics.rmse(),3)),
                 diff(list(X_test.columns.values), features[i])]
                )
    out = pd.DataFrame(results)
    out.to_csv(path, header=False)
    sys.stdout.write('\r' + ' - Results are stored in: ' + path+'\n')
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also

    #print(out.to_string(header=None,index=None))
    return out


def fullname(o):
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = o.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return o.__class__.__name__  # Avoid reporting __builtin__
    else:
        return module + '.' + o.__class__.__name__

def convertMillis(millis):
    milliseconds = millis%1000
    seconds = int(millis / 1000) % 60
    minutes = int(millis / (1000 * 60)) % 60
    hours = int(millis / (1000 * 60 * 60))
    return str(hours)+':'+str(minutes)+':'+str(seconds)+'.'+str(milliseconds)
