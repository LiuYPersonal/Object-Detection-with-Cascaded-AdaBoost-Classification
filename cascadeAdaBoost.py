import numpy as np
import cv2
#Cascaded AdaBoost Algorithm
def AdaBoost(features,weight,label,N_features,N_pos,N_neg,stage):
    T=50
    feature_idxs=[]
    thetas=[]
    polars=[]
    results=[]
    alphas=[]
    zero_err_flag=False
    #iterate to find weak classifier
    for t in range(T):
        print 'T:   ',t
        min_err=np.inf
        
        #normalize weight
        weight=weight/np.sum(weight)
        
        #calculate pos weight
        t_pos=np.sum(weight[:N_pos])
        print 't_pos',t_pos
        t_neg=1.0-t_pos
        count=0
        # Select best weak classifier with respect to the weighted error
        for feature in features:
            err=np.zeros((len(feature),2))           
            sorted_idx=np.argsort(feature)
            sorted_weight=weight[sorted_idx]
            sorted_label=label[sorted_idx]
            feature=np.sort(feature)
            
            #compute error
            s_pos=np.cumsum(sorted_weight*sorted_label)
            s_neg=np.cumsum(sorted_weight)-s_pos
                        
            err[:,0]=s_pos+(t_neg-s_neg)
            err[:,1]=s_neg+(t_pos-s_pos)
            #find smallest error
            e=err.min(1)
            current_err=e.min()
          
            #select threshold for the smallest error
            err_idx=e.argmin()
            #find the best weak classifier
            if current_err< min_err and current_err > 0.0:
                result=np.zeros((N_pos+N_neg),dtype='int')
                result_tmp=np.zeros((N_pos+N_neg),dtype='int')
                #store minimun err
                min_err=current_err
                feature_idx=count
                print min_err, err_idx
                #calculate result and polarity
                if err[err_idx,0] <= err[err_idx,1]:
                    p=-1
                    result_tmp[err_idx:]=1
                    for k in range(N_pos+N_neg):
                        result[sorted_idx[k]]=result_tmp[k]
                else:
                    p=1
                    result_tmp[:err_idx]=1
                    for k in range(N_pos+N_neg):
                        result[sorted_idx[k]]=result_tmp[k]
                    
                #calculate theta
                if err_idx==0:
                    theta=feature[0]-0.5
                elif err_idx==len(feature)-1:
                    theta=feature[err_idx]+0.5
                else:
                    theta=(feature[err_idx]+feature[err_idx+1])/2
            count+=1

        #save feature id, polarity, thresholds, errors and result for future use
        feature_idxs.append(feature_idx)
        polars.append(p)
        thetas.append(theta)
        results.append(result)
        
        # calculate alpha
        # special case if error rate is 0
        beta=min_err/(1-min_err)
        alpha=np.log(1/beta)
        alphas.append(alpha)
        
        #calculate strong classifier if FP<0.5 stop
        strong_classifier=np.dot(alphas,results)
        print 'maximum of strong_classifier', max(strong_classifier)
        #make sure TP=1
        threshold=min(strong_classifier[:N_pos])
        print 'threshold',threshold
        C = (strong_classifier>=threshold) + 0
        print 'C', np.sum(C)
        FP=float(np.sum(C[N_pos:]))/float(N_neg)
        print "FP",FP
        if FP<0.5:
            break
        #classify error
        label_right=(result.astype(int)==label) + 0
        #update weight
        for i in range(len(weight)):
            if label_right[i]==1:
                weight[i]=weight[i]*beta
    
    np.savetxt('adaBoost_test/feature_idx'+str(stage)+'.txt',feature_idxs,fmt='%i')
    np.savetxt('adaBoost_test/polars'+str(stage)+'.txt',polars,fmt='%i')
    np.savetxt('adaBoost_test/thetas'+str(stage)+'.txt',thetas)
    np.savetxt('adaBoost_test/alphas'+str(stage)+'.txt',alphas)    
    return C
#feature_idxs,polars,thetas,results
def cascade_AdaBoost(filepath,N_pos,N_neg,Stage,N_features):
    FPs=[]
    #initialize index
    idx=np.arange((N_pos+N_neg))
    #initialize labels
    labels=np.ones((N_pos+N_neg),dtype='int')
    labels[N_pos:]=0
    #load data
    N_total=N_pos+N_neg
    print 'loading feature: '
    features=np.loadtxt(filepath,dtype='int',delimiter=',')
    print np.shape(features)
    #iterate stages
    for i in range(Stage):
        print 'Stage:', str(i)
        label=labels[idx]
        feature=features[:,idx]
        print 'N_neg:  ',N_neg
        print 'N_pos:  ',N_pos
        print 'Sum of label:  ', np.sum(label)
        #Uniformly initialize weight
        weight=np.zeros(len(idx))
        weight[:N_pos]=0.5/float(N_pos)
        weight[N_pos:]=0.5/float(N_neg)
        C=AdaBoost(feature,weight,label,N_features,N_pos,N_neg,i)
        print np.sum(C)
        #update indexes-eliminate negatives
        N_neg=np.sum(C[N_pos:])
        idx_new=np.arange((N_pos+N_neg),dtype='int')
        if N_neg==0:
            break
        # Reject all the false points
        idx=idx[C==1]
        FP=float(N_neg)/1758.0
        print 'FP: ', FP
        FPs.append(FP)
    np.savetxt('FP_train.txt',FPs)
    return i

#set params
N_pos=710
N_neg=1758
N_features=166000
filepath='train/features_train.txt'
##training classifier
Stage = 10
stage=cascade_AdaBoost(filepath,N_pos,N_neg,Stage,N_features)


