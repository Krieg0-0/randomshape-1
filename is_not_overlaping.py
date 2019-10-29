#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:53:27 2019

@author: csl
"""

import matplotlib.pyplot as plt
import numpy as np
import math
cos = math.cos
sin = math.sin
#n = 20
#N = 72 * n  #像素点个数为（72*n）^2
N = 720
pi = 3.14
M = 256     #圆边上取点的个数/2，基于参数方程
T = 2 * M   #圆边上取点的个数
k = 0       #需要生成的图形数
mm = 0      #尝试生成的图形数
angle = np.arange(0,2*pi,pi/M) #取点所需的角度
flag_1 = False
fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})#生成figure与axis
w, h = N, N
Matrix = [[0 for x in range(w)] for y in range(h)] #判断每个像素是否被填充的矩阵，默认为0
Matrix = np.array(Matrix)  #不加就报错
P = 1000
s = 200000/P


while k < P:#生成的图形数
    R = 2 * pi * np.random.random()
    x = np.random.randint(0,N) #圆心的横坐标
    y = np.random.randint(0,N) #圆心的纵坐标
    p = (1 - np.random.random())*(0.8) + 0.2
#    rr = np.random.randint(N/128,N/32) #半径
    r1 = int((s/p/pi)**(0.5))
    r2 = int((s*p/pi)**(0.5))
    
    
    
    xx = np.arange(T-1)
    yy = np.arange(T-1)
    
    for i in range(T-1): #给每个点的坐标赋值
#        xx[i] = x + rr*math.cos(angle[i]) T=64 xx:0-63   0-63
#        yy[i] = y + rr*math.sin(angle[i]) 0-31 63-32   
        xx[i] = int(r1*cos(angle[i])*cos(R)-r2*sin(angle[i])*sin(R) + x)
        yy[i] = int(r1*cos(angle[i])*sin(R)+r2*sin(angle[i])*cos(R) + y)
        
    kk = 0
    for b in range(T-1):
        for c in range(min(yy[T-2-b],yy[b]),max(yy[T-2-b],yy[b])):
            kk = kk + 1
    pan = [[0 for xx in range(2)] for yy in range(kk)]
    pan = np.array(pan)
    
    kkk = 0
    for bb in range(T-1):
        for cc in range(min(yy[T-bb-2],yy[bb]),max(yy[T-bb-2],yy[bb])):
            pan[kkk,0] = xx[bb]
            pan[kkk,1] = cc
            kkk = kkk + 1
    if len(pan) !=0:    
        pan = np.unique(pan,axis=0) 
    else:
        continue   
    if (max(xx) >= N) | (max(yy) >= N) | (min(xx) <= 0) | (min(yy) <= 0):
            continue 
    mm = mm + 1    
        
    for zz in range(len(pan)):
        if Matrix[pan[zz][0],pan[zz][1]] == 1:
            flag_1 = True
            for m in range(zz):
                Matrix[pan[m][0],pan[m][1]] = Matrix[pan[m][0],pan[m][1]] - 1 
            break
        else:
            flag_1 = False
            Matrix[pan[zz][0],pan[zz][1]] = Matrix[pan[zz][0],pan[zz][1]] + 1
            
    if flag_1 == False:
        ax.fill(xx,yy)
        print('尝试次数%d'%mm)
        k = k + 1
        print('生成次数%d'%k)      
#    for p in range(x-rr,x+rr)  :#获得围成该图形的最小矩形的“是否填充”的矩阵
#        for q in range(y-rr,y+rr):
#            
#            
#            if(Matrix[p,q] == 1):#如果有已被填充的，跳出p、q的循环
#                flag_1 = True #用于跳出两个循环
#                for m in range(x-rr,p):
#                    for n in range(y-rr,q):
#                        Matrix[m,n] = Matrix[m,n] - 1
#                break
#            else:
#                flag_1 = False 
#                Matrix[p,q] = Matrix[p,q] + 1 #如果没被填充，则在填充矩阵里记录
#        if flag_1:
#            break
# 
#    if  flag_1 == False:
#        ax.fill(xx,yy) #如果整个图形都未被填充，则描绘它 
#        print('尝试次数%d'%mm)
#        print('生成次数%d'%(k+1))
#        k = k + 1
#        
ax.set_xlim(0, N)
ax.set_ylim(0, N)
plt.savefig('test_test.svg')
plt.show()
    
