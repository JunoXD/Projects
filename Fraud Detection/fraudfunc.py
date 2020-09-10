import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import time
import multiprocessing
import itertools as it

def fdr_score(y_pred_prob, y_obs, alpha=0.03):
    """Calculate the FDR score at the given rejection rate of alpha"""
    num=int(round(alpha*len(y_obs)))
    threshold=min(np.quantile(y_pred_prob,1-alpha),y_pred_prob.max())
    cutoff=y_pred_prob[y_pred_prob>=threshold].min()
    above=y_obs[y_pred_prob>cutoff]
    more=num-len(above)
    tie=y_obs[y_pred_prob==cutoff]
    add=resample(tie,replace=False,n_samples=min(more,len(tie)))
    score=(sum(above)+sum(add))/sum(y_obs)
    return score

def trn_resample(X, y, good_bad_ratio=10):
    """Downsample the majority class (good) in the training set to match the specified good_bad_ratio. \
Returns X_downsampled, y_downsampled."""
    df = pd.concat([X, y], axis=1)
    goods = df[y==0]
    bads = df[y==1]
    n_bads = bads.shape[0]
    goods_downsampled = resample(goods,
                               replace=False,
                               n_samples=n_bads*good_bad_ratio)
    df_downsampled = pd.concat([goods_downsampled, bads])
    X_downsampled = df_downsampled.iloc[:,:-1].values
    y_downsampled = df_downsampled.iloc[:,-1].values
    return X_downsampled, y_downsampled

def fdr_summary(model, X_mod, y_mod, X_oot, y_oot, n_iter=10, test_size=0.3):
    """Random split modeling data multiple times, fit on train, and calculate FDR on train, test and oot sets.\
Returns a summary dataframe of FDRs on all sets during each iteration and the mean FDR on each set."""
    fdr_trainList = []
    fdr_testList = []
    fdr_ootList = []
    for i in range(n_iter):
        X_train,X_test,y_train,y_test=train_test_split(X_mod,y_mod,test_size=test_size,stratify=y_mod)
        X_train_ds, y_train_ds = trn_resample(X_train, y_train)
        model.fit(X_train_ds, y_train_ds)

        y_train_pred_prob = model.predict_proba(X_train)[:,1]
        y_test_pred_prob = model.predict_proba(X_test)[:,1]
        y_oot_pred_prob = model.predict_proba(X_oot)[:,1]

        fdr_trainList.append(fdr_score(y_train_pred_prob,y_train))
        fdr_testList.append(fdr_score(y_test_pred_prob,y_test))
        fdr_ootList.append(fdr_score(y_oot_pred_prob,y_oot))
    summary = pd.DataFrame([fdr_trainList, fdr_testList, fdr_ootList])
    summary.columns=[i for i in range(1,n_iter+1)]
    summary.index=['trn','tst','oot']
    summary['mean']=summary.mean(axis=1)
    summary=summary.T
    return summary

def avg_tst_fdr(model, X, y, n_iter=10, test_size=0.3):
    """Returns an average FDR on test set through all iterations."""
    fdr_tot=0
    for i in range(n_iter):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_size,stratify=y)
        X_train_ds, y_train_ds = trn_resample(X_train, y_train)
        model.fit(X_train_ds, y_train_ds)
        y_test_pred_prob = model.predict_proba(X_test)[:,1]
        fdr_tot+=fdr_score(y_test_pred_prob,y_test)
    return fdr_tot/n_iter

def get_paramsList(params_grid):
    """Create all possible combinations of params. \
Returns a list of all param names and a list of all param combinations."""
    allNames = sorted(params_grid)
    combinations = it.product(*(params_grid[Name] for Name in allNames))
    all_params = list(combinations)
    return allNames, all_params

def split(a, n):
    """Split a list into eqal parts. Returns a list of lists."""
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def main_func(model, X, y, param_names, all_params, split_round, test_size, verbose, fdr_dict):
    """Finds the param combination in the list of combinations that yields the highest avg FDR on the test set."""
    max_fdr = 0
    best_params = 0
    for cur_params in all_params:
        starttime = time.time()
        params = dict(zip(param_names, cur_params))
        if verbose:
            print(params)
        model.set_params(**params)
        fdr = avg_tst_fdr(model, X, y, n_iter=split_round, test_size=test_size)*100
        if verbose:
            print('{}, FDR: {:.2f}%, {} sec elapsed'.format(params, fdr, round(time.time() - starttime)))
        #print('fdr:', fdr)
        if fdr > max_fdr:
            max_fdr = fdr
            best_params = params
    fdr_dict[max_fdr] = best_params

def randomized_search(model, params_range, X, y, n_iter='auto', split_round=10, test_size=0.3, n_jobs=1, verbose=True):
    """Performs randomized search through the given params range and find the param combination that yields the \
highest avg FDR on the test set."""
    tot_starttime = time.time()

    param_names, all_params = get_paramsList(params_range)
    print("Total combination:", len(all_params))
    if n_iter=='auto':
        N = int(len(all_params)/10)
    else:
        N = n_iter
    print("Randomized search size:", N)

    selected = []
    for i in range(N):
        taken_params = random.choice(all_params)
        all_params.remove(taken_params)
        selected.append(taken_params)

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    paramsList = list(split(selected, n_jobs))

    manager = multiprocessing.Manager()
    fdr_dict = manager.dict()
    processes = []
    for i in range(n_jobs):
        p = multiprocessing.Process(target=main_func,
                                    args=(model, X, y, param_names, paramsList[i],
                                          split_round, test_size, verbose, fdr_dict))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    final_max_fdr = max(fdr_dict.keys())
    final_best_params = fdr_dict[final_max_fdr]

    print('Total time elapsed: {:.1f} minutes'.format((time.time() - tot_starttime)/60))

    return final_max_fdr, final_best_params

def grid_search(model, params_grid, X, y, split_round=10, test_size=0.3, n_jobs=1, verbose=True):
    """Performs grid search through the given params grid and find the param combination that yields the \
highest avg FDR on the test set."""
    tot_starttime = time.time()

    param_names, all_params = get_paramsList(params_grid)
    print("Total combination:", len(all_params))

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    paramsList = list(split(all_params, n_jobs))

    manager = multiprocessing.Manager()
    fdr_dict = manager.dict()
    processes = []
    for i in range(n_jobs):
        p = multiprocessing.Process(target=main_func,
                                    args=(model, X, y, param_names, paramsList[i],
                                          split_round, test_size, verbose, fdr_dict))
        processes.append(p)
        p.start()

    for process in processes:
        process.join()

    final_max_fdr = max(fdr_dict.keys())
    final_best_params = fdr_dict[final_max_fdr]

    print('Total time elapsed: {:.1f} minutes'.format((time.time() - tot_starttime)/60))

    return final_max_fdr, final_best_params
