from datetime import datetime as dt
from random import sample

import numpy as np
import pandas as pd

from methods import recall_score as rs, precision_score as ps, accuracy_score as acs
from methods import confusion_matrix as cm
from resultlib import DUMP_OUT, FAMILY_OUT
import sourcelib



def create_train_sample(data, f, length = 10000):
    '''
    '''

    gp = data.head(50000).groupby(data.family == f)
    data_train = gp.get_group(True).append(gp.get_group(False).sample(length - gp.groups[True].size), ignore_index = True)
    return data_train 


def process(data, family = None, features = ['a', 'e', 'sinI'], algorithm = None, *, dump_out = False, verbose = 1):
    '''
    
    '''

    assert family is not None and family > 0, "The family number must be defined"
    assert algorithm is not None, "The ML classifier must be defined"
    
    if verbose >=1:
        print('Process: family = ', family)

    # Create train sample
    data_train = create_train_sample(data, family)

    xall = data[features].values
    yall = np.array([1 if int(i) == family else 0 for i in data.family])
    xtrain = data_train[features].values
    ytrain = np.array([1 if int(i) == family else 0 for i in data_train.family])

    if verbose >= 1:
        nowstart = dt.now()
        print('Start time: ', nowstart, "\n")

    # Fit the algorithm
    alg = algorithm
    alg.fit(xtrain, ytrain)
    yallnew = np.array([])

    start_item = 0
    end_item = len(data)
    i2 = start_item

    if dump_out:
        dumpout = DUMP_OUT 
        dumpout.insert(1, str(f))
        dumpout = ''.join(dumpout)
        dump_file = open(dumpout, 'w')

    # Start processing with small dumpable blocks
    while True:
        i1 = i2
        i2 = min(i1 + 10000, end_item)
        xtest = xall[i1:i2]
        ytest = yall[i1:i2]
        m = 0
        ylocalnew = np.array([])

        if verbose >= 2:
            print('Processing block from ', i1, ' to ', i2)
            nowlocal = dt.now()

        # Processung local subblocks
        while m < i2 - i1:
            m2 = min(m + 500, i2 - i1)
            scr = alg.predict(xtest[m:m2])
            ylocalnew = np.concatenate((ylocalnew, scr))

            if verbose >= 3:
                print(' ------ ', m, m2, ' ------ ', dt.now())
            m = m2

        # Local scores
        if verbose >= 3:
            print('Local scores - precision: ', ps(ytest, ylocalnew), ', recall: ',
                  rs(ytest, ylocalnew), ', accuracy: ', acs(ytest, ylocalnew))
            print(cm(ytest, ylocalnew))

        if verbose >= 2:
            print('Finished block from ', i1, ' to ', i2)
            print('Local time consumed: ', (dt.now() - nowlocal).total_seconds() , ' s.\n')

        yallnew=np.concatenate((yallnew, ylocalnew))

        # Dumping results
        if dump_out:
            for j in range(i1, i2):
                print(' '.join(map(str, data.loc[j])),
                    yall[j],
                    yallnew[j - t1],
                    file = dump_file)
                dump_file.flush()

        if i2 >= end_item: break

    if dump_out:
        dump_file.close()

    # Total scores
    total_precision = ps(yall[start_item:end_item], yallnew)
    total_recall = rs(yall[start_item:end_item], yallnew)
    total_accuracy = acs(yall[start_item:end_item], yallnew)
    total_cmatrix = cm(yall[start_item:end_item], yallnew)

    if verbose >= 2:
        print('Algorithm: ', repr(alg))

    if verbose >= 1:
        print('Total precision: ', total_precision,
              ', recall: ', total_recall,
              ', accuracy: ',total_accuracy,
              ', confusion matrix:\n')
        print(cm(yall[start_item:end_item], yallnew))
        nowend = dt.now()
        print('Finish time: ', nowend)
        print('Total time consumed: ', (nowend - nowstart).total_seconds(),"s.\n")
        if dump_out:
            print('The results have been dumped to: ./', dumpout)

    return total_precision, total_recall, total_accuracy, total_cmatrix


def custom_process(data, first = None, last = None, algorithm = None, varp = None, n_samples = 10, dump_out = True, verbose = 1):

    def _multiset(d):
        '''Yields cartesian product elements from a dictionary containing sets of variable parameters'''

        ld = len(d)
        lk = list(d)
        lenk = [len(d[i]) for i in lk]
        its=[0] * ld
        fullnum = np.cumprod(lenk)[-1]
        for i in range(fullnum):
            yield tuple( d[lk[k]][its[k]] for k in range(ld) )
            its[-1] += 1
            for j in range(ld - 1, 0, -1):
                if its[j] == lenk[j]:
                    its[j - 1] += 1
                    its[j] = 0

    fam_list = sourcelib.FAMILY_NUMBERS
    assert algorithm is not None, "The ML classifier must be defined"
    assert first in fam_list, 'Please specify correct family numbers'
    if last is None:
        last = first 
    if last not in fam_list:
        print('Warning: the last family number is not in a set of possible family numbers and only first family will be processed.')
        last = first 

    if verbose >= 1:
        nowstart = dt.now()
        print('Start processing using algorithm:', repr(algorithm))
        print('Start time: ', nowstart, "\n")

    index_first = fam_list.index(first)
    index_last = fam_list.index(last) + 1

    varp_fam = varp.copy()
    varp_fam.update({'family': fam_list[index_first:index_last]})
    res_indexcolumn = pd.MultiIndex.from_tuples(list(_multiset(varp_fam)), names = list(varp_fam))

    varpkeys = list(varp)
    cols = ['size', 'tp', 'tn', 'fp', 'fn', 'accuracy', 'precision', 'recall']
    config_list = pd.MultiIndex.from_tuples(list(_multiset(varp)), names = varpkeys)

    res = pd.DataFrame(index = res_indexcolumn, columns = cols)
    res = res.reorder_levels(['family'] + list(varp.keys())).sort_index()

    if dump_out:
        dumpout = FAMILY_OUT 
        dumpout.insert(2, str(last))
        dumpout.insert(1, str(first))
        dumpout = ''.join(dumpout)
        dump_file = open(dumpout, 'w')

    for i in fam_list[index_first:index_last]:

        if verbose >= 2:
            print('Processing family: ', i)

        for j in config_list:
            alg_config = {varpkey: j[k] for k, varpkey in enumerate(varpkeys)}

            if verbose >= 3:
                print('Configuration:\n', alg_config)

            [p, r, a, c] = list(zip(*tuple(process(data, family = i, algorithm = algorithm(**alg_config), verbose = verbose - 3) for s in range(n_samples))))

            # TODO: make shorter with "zip/map/lambda"
            d = {
                'size': sum([k[1][0] + k[1][1] for k in c]) / n_samples,
                'tp': sum(k[1][1] for k in c) / n_samples,
                'tn': sum(k[0][0] for k in c) / n_samples,
                'fp': sum(k[0][1] for k in c) / n_samples,
                'fn': sum(k[1][0] for k in c) / n_samples,
                'accuracy': sum(a) / n_samples,
                'precision': sum(p) / n_samples,
                'recall': sum(r) / n_samples}

            res.loc[tuple([i] + list(j))] = pd.Series(d)

    if dump_out:
        res.to_csv(dumpout) #,index=False) # TODO - check whether this option is needed
        if verbose >= 1:
            print('The results have been dumped to: ./', dumpout)

    if verbose >= 1:
        nowend = dt.now()
        print('Finish time: ', nowend)
        print('Total time consumed: ', (nowend - nowstart).total_seconds(),"s.\n")

    return res

