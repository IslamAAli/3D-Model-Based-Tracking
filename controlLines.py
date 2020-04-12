#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 04:20:07 2020

@author: junaid_ia
"""
import math
import numpy as np
def controlLines(cp):
    """RANSAC algorithm for picking up the best lines of an object"""
    n=10#number of picked at random
    k=0 #Number of iterations
    t=3 #Thershold for the distance from the lines
    lines=np.zeros((20,2))
    d=10#These values can be tuned
    i=0
    while(k==20):
        np.random.shuffle(cp)
        train = cp[1:n,:]
        val = cp[n:,:]
        train = np.hstack((train, np.ones((train.shape[0],1))))
        val = np.hstack((val, np.ones((val.shape[0],1))))
        X=train[:,[0,2]]
        Y=train[:,[1]]
        lines[i,:]=np.dot(np.linalg.pinv(X),Y).T
        Xt=val[:,[0,2]]
        Yt=val[:,1]
        dist=(Yt - np.dot(Xt,lines[k,:].T))/math.sqrt(1+(lines[k,1]**2)) 
        val=val[dist>t]
        if val.shape[0] >= d:
            lines[i,:]=np.dot(np.linalg.pinv(Xt),Yt).T
            i+=1
        k+=1
    return lines