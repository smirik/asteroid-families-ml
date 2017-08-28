import numpy as np
import sklearn as sl
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn.metrics import recall_score as rs, precision_score as ps, accuracy_score as acs
from sklearn.metrics import confusion_matrix as cm
from datetime import datetime as dt

def ro(x,y, k1=1.25, k2=2.0, k3=2.0):
    return (x[1]+y[1])*(x[4]+y[4])/4.0*np.sqrt(
        4.0*k1*((x[1]-y[1])/(x[1]+y[1]))**2+
        k2*(x[2]-y[2])**2+k3*(x[3]-y[3])**2)

def fun(nmax=406251):
    train={}
    count=0
    for s1,s2 in zip(open('all_tro.txt','r'),open('synt.csv','r')):
        count += 1
        i1=s1.rstrip().split()
        i2=s2.rstrip().split()
        src={'id': int(i1[0]),
            'fam':int(i1[3]),
            'number': int(i2[0]),
            'a':float(i2[2]),
            'e':float(i2[3]),
            'i':np.arcsin(float(i2[4])),
            'n':float(i2[5]),
            }
        train[src['id']] = src
        if count==nmax:
            break
    return train

def process(how_many_do_you_need=20000):
    print('Process')
    f=221
    k=5

    t=fun(nmax=how_many_do_you_need)
    xall=np.array([[t[i]['id'],t[i]['a'],t[i]['e'],t[i]['i'],t[i]['n'],t[i]['fam']] for i in t])
    yall=np.array([1 if int(i[5])==f else 0 for i in xall])
    xlen=max(10000,min(406251,how_many_do_you_need))
    xtrain=xall[:10000]
    ytrain=np.array([1 if int(i[5])==f else 0 for i in xtrain])

    print(dt.now(),"\n")

    cl=knc(n_neighbors=k,metric=ro,n_jobs=6)
    cl.fit(xtrain,ytrain)
    #for f in [3,4,5,10,15,20,24,158,170,221,8,43]:
    i=21
    yallnew=np.array([])
    while True:
        i1=i*10000
        i2=min((i+1)*10000,xlen)
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
        #ynew=cl.predict(xtest)
        print('i1: ',i1,', i2: ',i2,', K: ',k,', fam',f,', precision: ',ps(ytest,ynew),', recall: ',rs(ytest,ynew))
        print(dt.now(),"\n")
        yallnew=np.concatenate((yallnew,ynew))
        fi=open('dump.txt','a')
        for j in range(i1,i2):
            print(xall[j][0],
                xall[j][1],
                xall[j][2],
                xall[j][3],
                xall[j][4],
                xall[j][5],
                yall[j],
                yallnew[j-210000],
                file=fi)
        fi.close()
        if i2>=xlen: break
        i+=1
    #yallnew=np.array(yallnew)
    print('Total precision: ',ps(yall[210000:xlen],yallnew),', recall: ',rs(yall[210000:xlen],yallnew))
    print(dt.now(),"\n")
    print('Done.')

    return None
