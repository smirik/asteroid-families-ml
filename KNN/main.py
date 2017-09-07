import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import recall_score as rs, precision_score as ps, accuracy_score as acs
from sklearn.metrics import confusion_matrix as cm
from datetime import datetime as dt

##def ro(x,y, k1=1.25, k2=2.0, k3=2.0):
    ##return (x[1]+y[1])*(x[4]+y[4])/4.0*np.sqrt(
        ##4.0*k1*((x[1]-y[1])/(x[1]+y[1]))**2+
        ##k2*(x[2]-y[2])**2+k3*(x[3]-y[3])**2)

def ro(x,y, k1=1.25, k2=2.0, k3=2.0):
    return (x[0]+y[0])*(x[3]+y[3])/4.0*np.sqrt(
        4.0*k1*((x[0]-y[0])/(x[0]+y[0]))**2+
        k2*(x[1]-y[1])**2+k3*(x[2]-y[2])**2)

#==============================================================================================

def process_knn(f=None, train_sample=10000,t1=0,t2=9999,p=2,out=False,
            inp='data.csv',feats=['a','e','i'],k=5):

    if f==None: return (None,None,None,None)

    print('Process: f=',f)
    data = pd.read_csv(inp,sep=' ')
    gp=data.head(50000).groupby(data.fam==f)
    data_train = gp.get_group(True).append(gp.get_group(False).sample(train_sample-gp.groups[True].size),
        ignore_index=True)

    xall = data[feats].values
    yall=np.array([1 if int(i)==f else 0 for i in data.fam])
    xtrain=data_train[feats].values
    ytrain = np.array([1 if int(i)==f else 0 for i in data_train.fam])

    print('Start time: ',dt.now(),"\n")

    cl=knc(n_neighbors=k,metric="minkowski",p=p,n_jobs=6)
    cl.fit(xtrain,ytrain)
    i2=t1
    yallnew=np.array([])
    if out:
        fi=open('dump'+str(f)+'_minkow.txt','w')
    while True:
        i1=i2
        i2=min((i1+10000,t2+1))
        xtest=xall[i1:i2]
        ytest=yall[i1:i2]
        m=0
        ynew=np.array([])
        while m < i2-i1:
            m2=min(m+500,i2-i1)
            scr=cl.predict(xtest[m:m2])
            ynew=np.concatenate((ynew,scr))
            #print('--------- ',m,m2,' ------ ',dt.now())
            m=m2
        print('i1: ',i1,', i2: ',i2)#,', K: ',k,', fam',f,', precision: ',ps(ytest,ynew),', recall: ',
            #rs(ytest,ynew),', accuracy: ',acs(ytest,ynew))
        #print(cm(ytest,ynew))
        print(dt.now(),"\n")
        yallnew=np.concatenate((yallnew,ynew))
        if out:
            for j in range(i1,i2):
                print(' '.join(map(str,data.loc[j])),
                    yall[j],
                    yallnew[j-t1],
                    file=fi)
        if i2>=t2+1: break
    if out:
        fi.close()
    psa=ps(yall[t1:t2+1],yallnew)
    rsa=rs(yall[t1:t2+1],yallnew)
    acsa=acs(yall[t1:t2+1],yallnew)
    cma = cm(yall[t1:t2+1],yallnew)
    if out:
        print('Total precision: ',psa,', recall: ',rsa,
            ', accuracy: ',acsa)
        print(cm(yall[t1:t2+1],yallnew))
    print('Finish time: 'dt.now(),"\n")
    print('Done.')

    return (psa,rsa,acsa,cma)

#==============================================================================================
# Duplicate specially for Zappala metric
def zappala(f=None, train_sample=10000,t1=0,t2=9999,out=True,
            inp='data.csv',k=5):
    if f==None: return (None,None,None,None)

    print('Process: f=',f)
    data = pd.read_csv(inp,sep=' ')
    gp=data.head(50000).groupby(data.fam==f)
    data_train = gp.get_group(True).append(gp.get_group(False).sample(train_sample-gp.groups[True].size),
        ignore_index=True)
    
    feats=['a','e','i','n']

    xall = data[feats].values
    yall=np.array([1 if int(i)==f else 0 for i in data.fam])
    xtrain=data_train[feats].values
    ytrain = np.array([1 if int(i)==f else 0 for i in data_train.fam])

    print(dt.now(),"\n")

    cl=knc(n_neighbors=k,metric=ro,n_jobs=6)
    cl.fit(xtrain,ytrain)
    i2=t1
    yallnew=np.array([])
    if out:
        fi=open('dump'+str(f)+'.txt','a')
    while True:
        i1=i2
        i2=min((i1+10000,t2+1))
        xtest=xall[i1:i2]
        ytest=yall[i1:i2]
        m=0
        ynew=np.array([])
        while m < i2-i1:
            m2=min(m+500,i2-i1)
            scr=cl.predict(xtest[m:m2])
            ynew=np.concatenate((ynew,scr))
            print('--------- ',m,m2,' ------ ',dt.now())
            m=m2
        print('i1: ',i1,', i2: ',i2,', K: ',k,', fam',f,', precision: ',ps(ytest,ynew),', recall: ',
            rs(ytest,ynew),', accuracy: ',acs(ytest,ynew))
        print(cm(ytest,ynew))
        print(dt.now(),"\n")
        yallnew=np.concatenate((yallnew,ynew))
        if out:
            for j in range(i1,i2):
                print(' '.join(map(str,data.loc[j])),
                    yall[j],
                    yallnew[j-t1],
                    file=fi)
        if i2>=t2+1: break
    if out:
        fi.close()
    psa=ps(yall[t1:t2+1],yallnew)
    rsa=rs(yall[t1:t2+1],yallnew)
    acsa=acs(yall[t1:t2+1],yallnew)
    cma = cm(yall[t1:t2+1],yallnew)
    if out:
        print('Total precision: ',psa,', recall: ',rsa,
            ', accuracy: ',acsa)
        print(cm(yall[t1:t2+1],yallnew))
    print(dt.now(),"\n")
    print('Done.')

    return (psa,rsa,acsa,cma)

#==============================================================================================

def custom_process(first=None,n=None):
    
    l = [2,3,4,5,10,15,20,24,25,31,87,93,96,
        110,135,145,148,153,158,159,163,170,179,194,
        221,260,283,293,298,302,375,396,410,434,480,
        490,569,606,618,668,729,752,778,780,808,845,847,
        883,895,909,945,1040,1101,1118,1128,1189,1222,
        1298,1303,1338,1547,1658,1726,1911,2076,2782,
        3025,3330,3438,3460,3561,3811,3815,3827,4203,
        5026,5651,5931,6124,6355,6769,7468,7605,7744,
        8737]
    first = 0 if first==None else min(max(0,first),len(l))
    n = len(l) if n==None else min(max(0,n),len(l))
    d={}
    for i in l[first:n]:
        (p,r,a,c) = process_knn(f=i,t1=0,t2=406250)
        d[i]={'family':i,
            'size':c[1][0]+c[1][1],
            'tp':c[1][1],
            'tn':c[0][0],
            'fp':c[0][1],
            'fn':c[1][0],
            'accuracy':a,
            'precision':p,
            'recall':r}
    data=pd.DataFrame.from_dict(d,orient='index')[['family','size',
                                                   'tp','tn','fp','fn',
                                                   'accuracy','precision','recall']]
    data.to_csv('KNN_families_statistics.csv',index=False)
    
    return None

# NOT IN USE
#==============================================================================================
#def create_data():
    #fi=open('data.csv','w')
    #print('id a e i n lce fam',file=fi) 
    #for s1,s2 in zip(open('source/all_tro.txt','r'),open('source/synt.csv','r')):
        #i1=s1.rstrip().split()
        #i2=s2.rstrip().split()
        #src={'id': int(i1[0]),
            #'fam':int(i1[3]),
            #'number': int(i2[0]),
            #'a':float(i2[2]),
            #'e':float(i2[3]),
            #'i':np.arcsin(float(i2[4])),
            #'n':float(i2[5]),
            #'lce':float(i2[8])
            #}
        #print(src['id'],src['a'],src['e'],src['i'],src['n'],src['lce'],src['fam'],file=fi)
    #fi.close()
    #return None
