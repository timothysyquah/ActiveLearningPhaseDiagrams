#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 00:01:50 2020

@author: tquah
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from skimage import measure
#print(sklearn.__version__)
from sklearn.model_selection import KFold, GroupKFold, ShuffleSplit, StratifiedKFold,\
    GroupShuffleSplit, StratifiedShuffleSplit,train_test_split
#from sklearn.model_selection import Kfold
import pandas as pd
from skimage import measure
from scipy.interpolate import interp1d
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB,CategoricalNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from matplotlib.colors import ListedColormap
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

color  = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
marker = ('+', 'o', '*','v','^','<','>','s','p','h','H','x')

def spline_sort(df,boundary):
    x = df['X-'+boundary].to_numpy()
    y = df['Y-'+boundary].to_numpy()
    bool_nan = np.isnan(x)
    loc = np.where(bool_nan==False)[0]
    x = x[loc]
    y = y[loc]
    sort_y = np.argsort(y)
    x = x[sort_y]
    y = y[sort_y]
    cs  = interp1d(y,x)
    return [cs,np.array([np.min(x),np.max(x)]),np.array([np.min(y),np.max(y)])]
#print(coordinate_data)
def simple_phase_plotter(coordinate_data,zdata,xlabel = '$f_A$',ylabel = '$\chi N$'):
    fig = plt.figure()
    for i in np.unique(zdata):
        loc = np.where(i==zdata)[0]
        plt.scatter(coordinate_data[loc,0],coordinate_data[loc,1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.show()
    return fig


def normalize(coord,xmin,xmax,ymin,ymax):
    newcoord = np.zeros_like(coord)
    newcoord[:,0] = (coord[:,0]-xmin)/(xmax-xmin) 
    newcoord[:,1] = (coord[:,1]-ymin)/(ymax-ymin) 
    return newcoord
def invnormalize(normcoord,xmin,xmax,ymin,ymax):
    coord = np.zeros_like(normcoord)
    coord[:,0] = (xmax-xmin)*normcoord[:,0]+xmin
    coord[:,1] = (ymax-ymin)*normcoord[:,1]+ymin
    return coord
def decision_boundary_delaunay_plot(coords,zdata,phase_transitions=np.array([[0,1],[1,2],[2,3],[3,4]])):
    tri = Delaunay(coords)
    plt.figure()
    plt.triplot(coords[:,0],coords[:,1],tri.simplices,c = 'k',alpha = 0.2)
    # plt.scatter(coords[:,0],coords[:,1],c = 'r',s = 1.)
    
    zunique = np.unique(zdata)
    for i in range(0,len(zunique),1):
        zloc = np.where(zunique[i]==zdata)[0]
        plt.scatter(coords[zloc,0],coords[zloc,1],color = color[i],marker= marker[i],alpha = 0.5)
    
    
    trigrid = tri.simplices
    edges = [[0,1],[0,2],[1,2]]
    boundaries = []
    phase_number = np.sum(phase_transitions,axis=1)
    for boundary in phase_transitions:
        boundaries.append([])
    for triangle in trigrid:
        tricoords = coords[triangle,:]
        zvalues = zdata[triangle]
        for i in range(0,3,1):
            z_edge = zvalues[edges[i]]
            if z_edge[1]!=z_edge[0]:
                # print(z_edge)
                phaseloc = np.where(np.sum(z_edge)==phase_number)[0]
                if len(phaseloc)>0:
                    boundary_value = np.mean(tricoords[edges[i]],axis = 0)
                    boundaries[phaseloc[0]].append(boundary_value)
                # plt.plot(boundary_value[0],boundary_value[1],'og')
    boundary_return = []
    for boundary in boundaries:
        newarray = np.vstack(boundary)
        ysort = np.argsort(newarray[:,1])
        returnarray = newarray[ysort,:]
        # plt.plot(returnarray[:,0],returnarray[:,1])
        yunique = np.unique(returnarray[:,1])
        unique_boundary = np.zeros((len(yunique),2))
        for i in range(0,len(yunique)):
           unique_boundary[i,1]= yunique[i]
           loc = np.where(yunique[i]==returnarray[:,1])[0]
           unique_boundary[i,0]= np.mean(returnarray[loc,0])
        plt.plot(unique_boundary[:,0],unique_boundary[:,1])
        plt.xlabel('$f_A$')
        plt.ylabel('$\chi N$')
        boundary_return.append(unique_boundary)
    plt.savefig('Base_Method_1.png',dpi = 300)
    return boundary_return

def decision_boundary_delaunay(coords,zdata,phase_transitions=np.array([[0,1],[1,2],[2,3],[3,4]])):
    tri = Delaunay(coords)
    
    zunique = np.unique(zdata)
    for i in range(0,len(zunique),1):
        zloc = np.where(zunique[i]==zdata)[0]
    
    
    trigrid = tri.simplices
    edges = [[0,1],[0,2],[1,2]]
    boundaries = []
    phase_number = np.sum(phase_transitions,axis=1)
    for boundary in phase_transitions:
        boundaries.append([])
    for triangle in trigrid:
        tricoords = coords[triangle,:]
        zvalues = zdata[triangle]
        for i in range(0,3,1):
            z_edge = zvalues[edges[i]]
            if z_edge[1]!=z_edge[0]:
                # print(z_edge)
                phaseloc = np.where(np.sum(z_edge)==phase_number)[0]
                if len(phaseloc)>0:
                    boundary_value = np.mean(tricoords[edges[i]],axis = 0)
                    boundaries[phaseloc[0]].append(boundary_value)
                # plt.plot(boundary_value[0],boundary_value[1],'og')
    boundary_return = []
    for boundary in boundaries:
        if len(boundary)>0:
            newarray = np.vstack(boundary)
            ysort = np.argsort(newarray[:,1])
            returnarray = newarray[ysort,:]
            # plt.plot(returnarray[:,0],returnarray[:,1])
            yunique = np.unique(returnarray[:,1])
            unique_boundary = np.zeros((len(yunique),2))
            for i in range(0,len(yunique)):
               unique_boundary[i,1]= yunique[i]
               loc = np.where(yunique[i]==returnarray[:,1])[0]
               unique_boundary[i,0]= np.mean(returnarray[loc,0])
            boundary_return.append(unique_boundary)

    return boundary_return