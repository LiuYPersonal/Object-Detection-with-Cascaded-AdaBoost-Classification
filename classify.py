import numpy as np
import cv2
#classify testing data
def classify(features,filepath_classifier,stage):
    
    #load classifier
    feature_idxs=np.loadtxt(filepath_classifier+'feature_idx'+str(stage)+'.txt', dtype='int')
    polars=np.loadtxt(filepath_classifier+'polars'+str(stage)+'.txt')
    thetas=np.loadtxt(filepath_classifier+'thetas'+str(stage)+'.txt')
    alphas=np.loadtxt(filepath_classifier+'alphas'+str(stage)+'.txt')
    
    #compute weak classifier
    T=len(feature_idxs)
    results=[]
    for t in range(T):
        feature=features[feature_idxs[t]]
        result=(polars[t]*feature<polars[t]*thetas[t]) + 0
        results.append(result)
        
    #strong classifier
    strong_classifier=np.dot(alphas,results)
    threshold=.5*np.sum(alphas)
    final_result=(strong_classifier>=threshold) + 0

    return final_result

#set params
N_pos=888-710
N_neg=2198-1758

filepath_feature='train/features_train.txt'
filepath_classifier='adaBoost_test/'

#load data
print 'load features:'
features=np.loadtxt(filepath_feature,dtype='int',delimiter=',')

#test classifier for each stage
stage=9
total_FP=[]
total_FN=[]

for i in range(stage):
    print 'stage: ', i
    final_result = classify(features,filepath_classifier,i)
    #Compute FP rate
    FP=float(np.sum(final_result[:N_pos]))/float(N_pos)
    #Compute FN rate
    FN=1.0-float(np.sum(final_result[N_pos:]))/float(N_neg)
    total_FP.append(FP)
    total_FN.append(FN)
    print FP,FN

np.savetxt('FP.txt',total_FP)
np.savetxt('FN.txt',total_FN)

