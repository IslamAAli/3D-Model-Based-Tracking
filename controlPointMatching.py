#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:52:33 2020

@author: junaid_ia
"""
import math
import numpy as np
import operator
from matplotlib import pyplot as plt
def controlPointMatching(cp,edges,edge_tags):
    kx=1
    ky=1
    beta=[math.atan(ky/kx), -math.atan(ky/kx)]
    no_of_edges = np.unique(edge_tags).shape[0]
    matchingPoints=np.zeros((cp.shape))
    k=0
    for i in range(0,no_of_edges):
        edge_control_points=cp[edge_tags==i]
        for j in range(edge_control_points.shape[0]):
            r = edge_control_points[0,:]
            s = edge_control_points[-1,:]
            temp = abs(s-r)
            dist_sr = math.sqrt(temp[1]**2+temp[0]**2)
            cos_alpha = temp[0]/dist_sr
            sin_alpha = temp[1]/dist_sr
            theta = math.atan(sin_alpha/cos_alpha)*(180/np.pi)
            if theta < 0:
                theta=180+theta
            direction={'H':abs(90 - theta),'V':abs(90-abs(theta-90)),'UD':abs(90-abs(theta-135)),'LD':abs(90-abs(theta-45))}
            case=min(direction.items(), key=operator.itemgetter(1))[0]
            #Now calculating the horizontal distance
            if case=='H':
                pos=find_Edge(edges,edge_control_points[j,:],'H')
                if ifRedundant(pos,matchingPoints):
                    pos=[np.inf,np.inf]
                    l=np.inf
                else:
                    n=pos[0]-edge_control_points[j,0]
                    l=-1*abs(n)*kx*sin_alpha
            #Now calculating the Vertical distance
            if case=='V':
                pos=find_Edge(edges,edge_control_points[j,:],'V')
                if ifRedundant(pos,matchingPoints):
                    pos=[np.inf,np.inf]
                    l=np.inf
                else:
                    n=pos[1]-edge_control_points[j,1]
                    l=abs(n)*ky*cos_alpha
            #Now calculating the First Diagonal distance
            if case=='UD':
                pos=find_Edge(edges,edge_control_points[j,:],'UD')
                if ifRedundant(pos,matchingPoints):
                    pos=[np.inf,np.inf]
                    l=np.inf
                else:
                    n=pos[1]-edge_control_points[j,1]
                    l=abs(n)*((ky*cos_alpha) +(math.copysign(1,n))*(kx*sin_alpha))
            #Now calculating the Second Diagonal distance
            if case=='LD':
                pos=find_Edge(edges,edge_control_points[j,:],'LD')
                if ifRedundant(pos,matchingPoints):
                    pos=[np.inf,np.inf]
                    l=np.inf
                else:
                    n=pos[1]-edge_control_points[j,1]
                    l=abs(n)*((ky*cos_alpha) +(math.copysign(1,n))*(kx*sin_alpha))
            matchingPoints[k,:]=np.asarray(pos)
            k+=1
#        fig,ax=plt.subplots()
#        ax.imshow(edges,cmap='gray')
#        ax.scatter(edge_control_points[:,0],edge_control_points[:,1],s=5,lw=1,facecolor="none",edgecolor="red")
#        ax.scatter(matchingPoints[:,0],matchingPoints[:,1],s=5,lw=1,facecolor="none",edgecolor="blue")

    return matchingPoints

def find_Edge(edges,point,tag):
    x=int(point[0])
    y=int(point[1])
    c1=0
    c2=0
    pos=[x,y]
    if tag=='H':
        flag=0
        while(flag==0 and c1<100  and c2>-100):
            if(edges[y,x+c1]==255):
                flag=1
                c2=0
                pos=[x+c1,y]
                break
            elif(edges[y,x+c2]==255):
                flag=1
                c1=0
                pos=[x+c2,y]
                break
            else:
                c1+=1
                c2-=1
            pos=[np.inf,np.inf]
    elif tag=='V':
        flag=0
        while(flag==0 and c1<100  and c2>-100):
            if(edges[y+c1,x]==255):
                flag=1
                c2=0
                pos=[x,y+c1]
                break
            elif(edges[y+c2,x]==255):
                flag=1
                c1=0
                pos=[x,y+c2]
                break
            else:
                c1+=1
                c2-=1
            pos=[np.inf,np.inf]
                
    elif tag=='UD':
        flag=0
        while(flag==0 and c1<100  and c2>-100):
            if(edges[y+c1,x+c1]==255):
                flag=1
                c2=0
                pos=[x+c1,y+c1]
                break
            elif(edges[y+c2,x+c2]==255):
                flag=1
                c1=0
                pos=[x+c2,y+c2]
                break
            else:
                c1+=1
                c2-=1
            pos=[np.inf,np.inf]
    elif tag=='LD':
        flag=0
        while(flag==0 and c1<100  and c2>-100):
            if(edges[y+c1,x+c2]==255):
                flag=1
                c2=0
                pos=[x+c2,y+c1]
                break
            elif(edges[y+c2,x+c2]==255):
                flag=1
                c1=0
                pos=[x+c2,y+c2]
                break
            else:
                c1+=1
                c2-=1
            pos=[np.inf,np.inf]
    else:
        print('TAG not defined')
    return pos

def ifRedundant(pos,matching_points):
    temp=matching_points==pos
    c=temp[:,0]*temp[:,1]
    return np.any(c == True)