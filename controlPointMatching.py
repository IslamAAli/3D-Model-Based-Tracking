#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:52:33 2020

@author: junaid_ia
"""
import math
import numpy as np

def controlPointMatching(cp,edges,edge_tags):
    
    kx=1
    ky=1
    beta=[math.atan(kx/ky), -math.atan(kx/ky)]
    no_of_edges = np.unique(edge_tags).shape[0]
    matchingPoints=np.zeros((cp.shape))
    k=0
    for i in range(1,no_of_edges+1):
        edge_control_points=cp[edge_tags==i]
        for j in range(edge_control_points.shape[0]):
            r = edge_control_points[0,:]
            s = edge_control_points[-1,:]
            temp = s-r
            dist_sr = math.sqrt(temp[1]**2+temp[0]**2)
            cos_alpha = temp[0]/dist_sr
            sin_alpha = temp[1]/dist_sr
            #Now calculating the horizontal distance
            pos1=find_Edge(edges,edge_control_points[j,:],'H')
            n1=pos1[0]-edge_control_points[j,0]
            l1=-1*abs(n1)*kx*sin_alpha
            #Now calculating the Vertical distance
            pos2=find_Edge(edges,edge_control_points[j,:],'V')
            n2=pos2[1]-edge_control_points[j,1]
            l2=abs(n2)*ky*cos_alpha
            #Now calculating the First Diagonal distance
            pos3=find_Edge(edges,edge_control_points[j,:],'UD')
            n3=pos3[1]-edge_control_points[j,1]
            l3=abs(n3)*((ky*cos_alpha) +(math.copysign(1,n3))*(kx*sin_alpha))
            #Now calculating the Second Diagonal distance
            pos4=find_Edge(edges,edge_control_points[j,:],'LD')
            n4=pos4[1]-edge_control_points[j,1]
            l4=abs(n4)*((ky*cos_alpha) +(math.copysign(1,n3))*(kx*sin_alpha))
            dist=np.array([[pos1,pos2,pos3,pos4],[abs(l1),abs(l2),abs(l3),abs(l4)]]).T
            dist=dist[dist[:,1]==min(dist[:,1])]
            matchingPoints[k,:]=np.asarray(dist[0,0])
            k+=1
    return matchingPoints

def find_Edge(edges,point,tag):
    x=int(point[0])
    y=int(point[1])
    c1=0
    c2=0
    pos=[x,y]
    if tag=='H':
        flag=0
        while(flag==0 and c1<50  and c2>-50):
            if(edges[x+c1,y]==255):
                flag=1
                c2=0
                pos=[x+c1,y]
                break
            elif(edges[x+c2,y]==255):
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
        while(flag==0 and c1<50  and c2>-50):
            if(edges[x,y+c1]==255):
                flag=1
                c2=0
                pos=[x,y+c1]
                break
            elif(edges[x,y+c2]==255):
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
        while(flag==0 and c1<50  and c2>-50):
            if(edges[x+c1,y+c1]==255):
                flag=1
                c2=0
                pos=[x+c1,y+c1]
                break
            elif(edges[x+c2,y+c2]==255):
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
        while(flag==0 and c1<50  and c2>-50):
            if(edges[x+c2,y+c1]==255):
                flag=1
                c2=0
                pos=[x+c2,y+c1]
                break
            elif(edges[x+c2,y+c2]==255):
                flag=1
                c1=0
                pos=[x+c1,y+c2]
                break
            else:
                c1+=1
                c2-=1
            pos=[np.inf,np.inf]
    else:
        print('TAG not defined')
    return pos