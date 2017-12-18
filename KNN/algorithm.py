from datetime import datetime as dt
from random import sample

import numpy as np

from methods import recall_score as rs, precision_score as ps, accuracy_score as acs
from methods import confusion_matrix as cm


_DUMP_OUT = ['source/result_', '.csv']


def create_train_sample(data, f, length = 10000):
    '''
    '''

    gp = data.head(50000).groupby(data.fam == f)
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
    data_train = create_train_sample(data, f, length)

    xall = data[features].values
    yall = np.array([1 if int(i) == f else negative for i in data.family])
    xtrain = data_train[feats].values
    ytrain = np.array([1 if int(i)==f else negative for i in data_train.family])

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
        dumpout = _DUMP_OUT 
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
            scr = alf.predict(xtest[m:m2])
            ynew = np.concatenate((ynew, scr))

            if verbose >= 3:
                print(' ------ ', m, m2, ' ------ ', dt.now())
            m = m2

        # Local scores
        if verbose >= 3:
            print('Local scores - precision: ', ps(ytest, ynew), ', recall: ',
                  rs(ytest, ynew), ', accuracy: ', acs(ytest, ynew))
            print(cm(ytest, ynew))

        if verbose >= 2:
            print('Finished block from ', i1, ' to ', i2)
            print('Local time consumed: ', (dt.now() - nowlocal).total_seconds() , ' s.\n')

        yallnew=np.concatenate((yallnew, ynew))

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

