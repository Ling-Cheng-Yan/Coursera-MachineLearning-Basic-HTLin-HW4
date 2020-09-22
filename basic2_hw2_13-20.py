import numpy as np

def getData(path):
    with open(path,'r') as fr:
        rawData=fr.readlines()
    lenx=len(rawData)
    leny=len(rawData[0].strip().split())
    dataSet=np.zeros((lenx,leny))
    labelSet=np.zeros((lenx,1))
    for i in range(lenx):
        line=rawData[i].strip().split()
        dataSet[i,0]=1
        for j in range(leny-1):
            dataSet[i,j+1]=float(line[j])
        labelSet[i]=int(line[-1])
    return dataSet,labelSet

def cal(dataSet,labelSet,lam):
    return np.dot(np.dot(np.linalg.pinv(np.dot(dataSet.T,dataSet)+lam*np.eye(dataSet.shape[1])),dataSet.T),labelSet)

def err(dataSet,labelSet,w):
    scores=np.dot(dataSet,w)
    predits=np.where(scores>=0,1,-1)
    err=sum(predits!=labelSet)
    return err*1.0/predits.shape[0]

def getLam():
    lam=[2,1,0,-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
    for i in range(len(lam)):
        lam[i]=10**lam[i]
    return lam
if __name__=='__main__':
    dataSet,labelSet=getData('hw4_train.dat.txt')
    w=cal(dataSet,labelSet,10)
    E_in=err(dataSet,labelSet,w)
    print(E_in)
    dataTest,labelTest=getData('hw4_test.dat.txt')
    E_out=err(dataTest,labelTest,w)
    print(E_out)
    #以上是13的解
    lam=getLam()
    E_in_min=100
    E_out_min=0
    corrLambda=0
    for i in range(len(lam)):
        w=cal(dataSet,labelSet,lam[i])
        E_in=err(dataSet,labelSet,w)
        E_out = err(dataTest, labelTest, w)
        if E_in<E_in_min:
            E_in_min=E_in
            E_out_min=E_out
            corrLambda=lam[i]
    print(corrLambda,E_in_min,E_out_min)
    #以上是14題的解
    lam = getLam()
    E_in_min = 0
    E_out_min = 100
    corrLambda = 0
    for i in range(len(lam)):
        w = cal(dataSet, labelSet, lam[i])
        E_in = err(dataSet, labelSet, w)
        E_out = err(dataTest, labelTest, w)
        if E_out < E_out_min:
            E_in_min = E_in
            E_out_min = E_out
            corrLambda = lam[i]
    print(corrLambda, E_in_min, E_out_min)
    #以上是15的題解
    lam = getLam()
    E_in_min = 100
    E_out_min = 0
    E_val_min=0
    corrLambda = 0
    for i in range(len(lam)):
        w=cal(dataSet[:120],labelSet[:120],lam[i])
        E_in = err(dataSet[:120],labelSet[:120], w)
        E_val=err(dataSet[120:],labelSet[120:],w)
        E_out = err(dataTest, labelTest, w)
        if E_in < E_in_min:
            E_in_min = E_in
            E_out_min = E_out
            E_val_min=E_val
            corrLambda = lam[i]
    print(corrLambda, E_in_min, E_val_min,E_out_min)
    #以上是16的題解
    lam = getLam()
    E_in_min = 0
    E_out_min = 0
    E_val_min = 100
    corrLambda = 0
    for i in range(len(lam)):
        w = cal(dataSet[:120], labelSet[:120], lam[i])
        E_in = err(dataSet[:120], labelSet[:120], w)
        E_val = err(dataSet[120:], labelSet[120:], w)
        E_out = err(dataTest, labelTest, w)
        if E_val < E_val_min:
            E_in_min = E_in
            E_out_min = E_out
            E_val_min = E_val
            corrLambda = lam[i]
    print(corrLambda, E_in_min, E_val_min, E_out_min)
    #以上是17的題解
    w = cal(dataSet, labelSet, corrLambda)
    E_in = err(dataSet, labelSet, w)
    E_out = err(dataTest, labelTest, w)
    print(corrLambda, E_in,E_out)
    #以上是18的題解
    lam = getLam()
    E_cv_min=100
    corrLambda=0
    for i in range(len(lam)):
        E_cv=0
        for j in range(1,int(len(dataSet)/40)+1):
            dataVirtualSet=dataSet
            labelVirtualSet=labelSet
            dataVal=dataSet[(j-1)*40:j*40]
            labelVal=labelSet[(j-1)*40:j*40]
            mask = [True] * dataSet.shape[0]
            for k in range((j-1)*40,j*40):
                mask[k]=False
            dataVirtualSet=dataSet[mask,:]
            labelVirtualSet=labelSet[mask,:]
            w = cal(dataVirtualSet, labelVirtualSet, lam[i])
            E_cv += err(dataVal, labelVal, w)
        E_cv/=float(len(dataSet)/40)
        if E_cv<E_cv_min:
            E_cv_min=E_cv
            corrLambda=lam[i]
    print(corrLambda,E_cv_min)
    #以上是19的題解
    w = cal(dataSet, labelSet, corrLambda)
    E_in = err(dataSet, labelSet, w)
    E_out = err(dataTest, labelTest, w)
    print(E_in,E_out)