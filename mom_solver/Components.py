# -*- coding: utf-8 -*-
"""
Created on Thu Aug 09 11:38:50 2018

@author: 913
"""

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
import itertools
#import pickle

from scipy.sparse import coo_matrix
#from _myutils import 
import pandas as pds
from multiprocessing.dummy import Pool

# In[] Some common parameters
from Parameters import QuadRule, IncidentPar, SolverPar, CellPar
from _myutils import Cubature, Triangle

# In[] RWG Func

class RWGFunc(object):
    def __init__(self):
        pass
    def match(self, grids, trias):
        '''
        '''        
        # scaning all triangles for their areas
        areas = []
        for tria in trias:
            pointA = grids[tria[0]]
            pointB = grids[tria[1]]
            pointC = grids[tria[2]]
            area = Triangle([pointA,pointB,pointC]).area(pointA,pointB,pointC)         
            areas.append(area)

        # scanning all triangles for hRWG
        hRWGs = []
        for idd, tria in enumerate(trias):
            hRWGs.append([idd, 0, tria[1], tria[2]]) # 分别保存为三角形编号、自由点编号、对应边的两个节点编号（全局编号）
            hRWGs.append([idd, 1, tria[2], tria[0]]) 
            hRWGs.append([idd, 2, tria[0], tria[1]])

        # scaning all hRWG for edges
        edges = []
        for idd, hRWG in enumerate(hRWGs):
            mark = False
            for edge in edges:
                if (edge[0][0] == hRWG[2] and edge[0][1] == hRWG[3]) or (edge[0][0]==hRWG[3] and edge[0][1]==hRWG[2]):
                    mark = True
                    edge[1].append(idd) # 保存对应hrwg的编号
                    break
            if mark == False:
                edges.append([(hRWG[2],hRWG[3]),[idd,]]) # 保存边的两个节点编号、对应hrwg的编号

        # generating rwg function
        rwgs = []
        hrwgs = []
        k_temp = 0
        for idd, edge in enumerate(edges):
            if len(edge[1])==2:
                triaP = hRWGs[edge[1][0]][0] # +三角形的全局编号
                freeP = hRWGs[edge[1][0]][1] # +的自由点全局编号
                triaM = hRWGs[edge[1][1]][0] # -三角形的全局编号
                freeM =  hRWGs[edge[1][1]][1] # -的自由点全局编号
                edgeLength = np.linalg.norm(np.array(grids[edge[0][0]])-np.array(grids[edge[0][1]])) # 公共边的长度
                gridP = [grids[trias[triaP][0]],grids[trias[triaP][1]],grids[trias[triaP][2]]] # +三角形的三个节点
                gridM = [grids[trias[triaM][0]],grids[trias[triaM][1]],grids[trias[triaM][2]]] # -三角形的三个节点
                hrwgs.append([gridP,freeP,0.5*edgeLength/areas[triaP], triaP]) # hrwg保存了三角形的节点、自由点局部编号、hrwg函数系数、三角形的全局编号
                hrwgs.append([gridM,freeM,0.5*edgeLength/areas[triaM], triaM]) # hrwg保存了三角形的节点、自由点局部编号、hrwg函数系数、三角形的全局编号
                rwgs.append({"+":[gridP,freeP,0.5*edgeLength/areas[triaP], triaP,k_temp],"-":[gridM,freeM,0.5*edgeLength/areas[triaM], triaM, k_temp+1]}) # rwg分别保存了两个三角形的节点、自由点局部编号、rwg函数系数、三角形的全局编号
                k_temp = k_temp+2
        return [hrwgs,rwgs] # return hrwg and rwg


# In[]  Right hand term
class IncidentFunc(object):
    def __init__(self,k, \
                 k_direct = np.array([0,-1,0.]).reshape([1,3]), \
                 e_direct=np.array([0.,0,1]).reshape([1,3])):
        self.k = k
        self.k_direct = k_direct
        self.e_direct = e_direct
        self.e_amp = 1.0
    def planewave(self,r1):
        temp1 = scipy.exp(-1.j*self.k*np.sum(self.k_direct*r1,axis=-1))*self.e_amp
        return np.dot(temp1.reshape([-1,1]),self.e_direct)

# In[] Matrix Filling

class FillingMatrix_dgf_free(object):
    def __init__(self, wavenumber, grids, trias):
        self.grids = grids
        self.trias = trias
        self.k = wavenumber
        self.tempIncPar = IncidentPar()
        self.planewave = IncidentFunc(self.k, self.tempIncPar.k_direct, self.tempIncPar.e_direct).planewave
        self.aita = 377
        self.common_Cal()
        pass
    def common_Cal(self):
        try:
            self.d = [[self.grids[self.trias[xx][0]], self.grids[self.trias[xx][1]], self.grids[self.trias[xx][2]]] \
                  for xx in xrange(len(self.trias))]
            self.d = np.array(self.d) ## NofTria _3 _3
            areas_d = [Triangle([]).area(temp_d[0],temp_d[1],temp_d[2]) for temp_d in self.d]
            areas_d = np.array(areas_d)
            
            b_11 = Cubature(1,1) #out-interg
            b_21 = Cubature(2,1) #Far
            b_31 = Cubature(3,1) #Neighbor
            b_41 = Cubature(4,1) #Same
            self.r_quad = dict()
            self.w_quad = dict()
            
            self.r_quad['b_11'] = np.array([[b_11.point(self.d[ii],ind) for ind in xrange(b_11.numPoint())]\
                           for ii,cell in enumerate(self.d)])
            self.r_quad['b_21'] = np.array([[b_21.point(self.d[ii],ind) for ind in xrange(b_21.numPoint())]\
                           for ii,cell in enumerate(self.d)])
            self.r_quad['b_31'] = np.array([[b_31.point(self.d[ii],ind) for ind in xrange(b_31.numPoint())]\
                           for ii,cell in enumerate(self.d)])
            self.r_quad['b_41'] = np.array([[b_41.point(self.d[ii],ind) for ind in xrange(b_41.numPoint())]\
                           for ii,cell in enumerate(self.d)])
            
            self.w_quad['b_11'] = np.array([[b_11.weight(areas_d[ii],ind) for ind in xrange(b_11.numPoint())] \
                       for ii,cell in enumerate(self.d) ])
            self.w_quad['b_21'] = np.array([[b_21.weight(areas_d[ii],ind) for ind in xrange(b_21.numPoint())] \
                       for ii,cell in enumerate(self.d) ])
            self.w_quad['b_31'] = np.array([[b_31.weight(areas_d[ii],ind) for ind in xrange(b_31.numPoint())] \
                       for ii,cell in enumerate(self.d) ])
            self.w_quad['b_41'] = np.array([[b_41.weight(areas_d[ii],ind) for ind in xrange(b_41.numPoint())] \
                       for ii,cell in enumerate(self.d) ])
            pass
        except Exception as e:
            print e
            raise
    
    def changeIncDir(self,incPar):
        self.tempIncPar = incPar
        self.planewave = IncidentFunc(self.k, self.tempIncPar.k_direct, self.tempIncPar.e_direct).planewave 
    def getk_direct(self):
        return self.tempIncPar.k_direct

    def coef_equations(self):
        temp = np.array([self.k*self.aita, -self.aita/self.k])*1.j/4./scipy.pi
        return temp
    def fillblock_dgf_free( self, triasinD1, triasinD2, b_101, b_12, rwgs):
        try:
            # 分区三角形的节点
            d1 = self.d[triasinD1,:]
            d2 = self.d[triasinD2,:]
            
            # 收集高斯点
            num1 = b_101.numPoint()
            r1Group_12 = self.r_quad["b_%d%d"%(b_101.degree(),b_101.rule())]\
                                     [triasinD1]
            r1Group_12 = r1Group_12.reshape([-1,3])
            w1Group_12 = self.w_quad["b_%d%d"%(b_101.degree(),b_101.rule())]\
                                     [triasinD1]
            w1Group_12 = w1Group_12.reshape([-1])
            r1Group_12_find = [ii for ii,cell in enumerate(d1) \
                               for ind in xrange(num1)]
            r1Group_12_find = np.array(r1Group_12_find) # NofTria
            
            num2 = b_12.numPoint()
            r2Group_12 = self.r_quad["b_%d%d"%(b_12.degree(),b_12.rule())]\
                                     [triasinD2]
            r2Group_12 = r2Group_12.reshape([-1,3])            
            w2Group_12 = self.w_quad["b_%d%d"%(b_12.degree(),b_12.rule())]\
                                     [triasinD2]
            w2Group_12 = w2Group_12.reshape([-1])
            r2Group_12_find = [ii for ii,cell in enumerate(d2) \
                               for ind in xrange(num2)]
            r2Group_12_find = np.array(r2Group_12_find)# NofTria
           
            # 形成K矩阵
            K12 = [r1Group_12,r2Group_12]

            # 形成F矩阵
            hrwginD1_p = [rwg['+'] for rwg in rwgs[1] \
                          if rwg['+'][3] in triasinD1]     # 找到所有hrwg
            S_matrix_rowinD1_p = [ii for ii,rwg in enumerate(rwgs[1]) \
                                  if rwg['+'][3] in triasinD1]
            S_matrix_valuesinD1_p = [1.0 for rwg in rwgs[1] \
                                     if rwg['+'][3] in triasinD1]            
            hrwginD1_n = [rwg['-'] for rwg in rwgs[1] \
                          if rwg['-'][3] in triasinD1]
            S_matrix_rowinD1_n = [ii for ii,rwg in enumerate(rwgs[1]) \
                                  if rwg['-'][3] in triasinD1]
            S_matrix_valuesinD1_n = [-1.0 for rwg in rwgs[1] \
                                     if rwg['-'][3] in triasinD1]             
            hrwginD1 = hrwginD1_p+hrwginD1_n
            S_matrix_rowinD1 = S_matrix_rowinD1_p+S_matrix_rowinD1_n
            S_matrix_colinD1 = np.arange(len(hrwginD1))
            S_matrix_valuesinD1 = S_matrix_valuesinD1_p+S_matrix_valuesinD1_n            
            S_matrixinD1 = coo_matrix((S_matrix_valuesinD1,(S_matrix_rowinD1, S_matrix_colinD1)),shape=[len(rwgs[1]),len(hrwginD1)]) #形成hrwg-rwg的变换矩阵        
            
            Tria_Hrwg = np.array([ff[0] for ff in hrwginD1]) 
            
            FreePoint_Hrwg = np.array([ff[0][ff[1]] for ff in hrwginD1]) # Hrwg*3 -- 所有hrwg的自由节点
            Weight_Hrwg = np.array([ff[2] for ff in hrwginD1]).reshape([1,-1,1]) # Hrwg*3 -- 所有hrwg的权重
                 
            r1Group_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]]) # 3*Hrwg*r -- 积分点
            free_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]])  # 3*Hrwg*r -- 所有hrwg的自由节点
            for id_comp in xrange(3): # 将其进行meshgrid
                r1Group_Hrwg[id_comp,:,:],free_Hrwg[id_comp,:,:] = np.meshgrid(r1Group_12[:,id_comp], FreePoint_Hrwg[:,id_comp])
            hrwgfunc = Weight_Hrwg*(r1Group_Hrwg - free_Hrwg) # 所有hrwg的取值
            hrwgfunc_div = np.multiply(Weight_Hrwg.reshape([-1,1]),np.ones([1,r1Group_12.shape[0]]))*2.0 # 所有hrwg的散度值

            temp_check_r_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]]) # 检查hrwg的合理值，去掉一些支撑集外面的非零值
            temp_check_hrwg_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]]) 
            class temp0(object):
                def __init__(self):
                    pass
                def iter(self,id):
                    id_x,id_y = id
                    temp_check_r_tria[id_x,id_y,:,:], temp_check_hrwg_tria[id_x,id_y,:,:] \
                                 = np.meshgrid(d1[r1Group_12_find][:,id_x,id_y], Tria_Hrwg[:,id_x,id_y])
            map(temp0().iter, [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)])
            temp_check = (temp_check_r_tria == temp_check_hrwg_tria) # 判断三角形是否一致
            check_matrix = (np.sum(np.sum(temp_check,axis=0),axis=0) == 9) # 只有三个点，九个数值都一样才表明是一个三角形
            temp2 = np.zeros(hrwgfunc.shape) # 过滤不合理的取值
            for ind_comp in xrange(3):
                temp2[ind_comp,:,:] = check_matrix*hrwgfunc[ind_comp,:,:]
            F1 = temp2[0]*w1Group_12
            F2 = temp2[1]*w1Group_12
            F3 = temp2[2]*w1Group_12
            F_div = check_matrix*hrwgfunc_div*w1Group_12
                
            # 形成G矩阵
            hrwginD2_p = [rwg['+'] for rwg in rwgs[1] if rwg['+'][3] in triasinD2]     # 找到所有hrwg
            S_matrix_rowinD2_p = [ii for ii,rwg in enumerate(rwgs[1]) if rwg['+'][3] in triasinD2]
            S_matrix_valuesinD2_p = [1.0 for rwg in rwgs[1] if rwg['+'][3] in triasinD2]
            
            S_matrix_rowinD2_n = [ii for ii,rwg in enumerate(rwgs[1]) if rwg['-'][3] in triasinD2]
            S_matrix_valuesinD2_n = [-1.0 for rwg in rwgs[1] if rwg['-'][3] in triasinD2]
            hrwginD2_n = [rwg['-'] for rwg in rwgs[1] if rwg['-'][3] in triasinD2] 
            
            hrwginD2 = hrwginD2_p+hrwginD2_n
            S_matrix_rowinD2 = S_matrix_rowinD2_p+S_matrix_rowinD2_n
            S_matrix_colinD2 = np.arange(len(hrwginD2))
            S_matrix_valuesinD2 = S_matrix_valuesinD2_p+S_matrix_valuesinD2_n
            
            
            S_matrixinD2 = coo_matrix((S_matrix_valuesinD2,(S_matrix_rowinD2, S_matrix_colinD2)),shape=[len(rwgs[1]),len(hrwginD2)])
            
            
            Tria_Hrwg = np.array([ff[0] for ff in hrwginD2]) 
            
            FreePoint_Hrwg = np.array([ff[0][ff[1]] for ff in hrwginD2]) # Hrwg*3 -- 所有hrwg的自由节点
            Weight_Hrwg = np.array([ff[2] for ff in hrwginD2]).reshape([1,-1,1]) # Hrwg*3 -- 所有hrwg的权重
            
            r2Group_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]]) # 3*Hrwg*r -- 积分点
            free_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]])  # 3*Hrwg*r -- 所有hrwg的自由节点
            for id_comp in xrange(3): # 将其进行meshgrid
                r2Group_Hrwg[id_comp,:,:],free_Hrwg[id_comp,:,:] = np.meshgrid(r2Group_12[:,id_comp], FreePoint_Hrwg[:,id_comp])
            hrwgfunc = Weight_Hrwg*(r2Group_Hrwg - free_Hrwg) # 所有hrwg的取值
            hrwgfunc_div = np.multiply(Weight_Hrwg.reshape([-1,1]),np.ones([1,r2Group_12.shape[0]]))*2.0 # 所有hrwg的散度值

            temp_check_r_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]]) # 检查hrwg的合理值，去掉一些支撑集外面的非零值
            temp_check_hrwg_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]]) 
            class temp1(object):
                def __init__(self):
                    pass
                def iter(self,id):
                    id_x,id_y = id
                    temp_check_r_tria[id_x,id_y,:,:], temp_check_hrwg_tria[id_x,id_y,:,:] \
                                 = np.meshgrid(d2[r2Group_12_find][:,id_x,id_y], Tria_Hrwg[:,id_x,id_y])
            map(temp1().iter, [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)])
            temp_check = (temp_check_r_tria == temp_check_hrwg_tria) # 判断三角形是否一致
            check_matrix = (np.sum(np.sum(temp_check,axis=0),axis=0) == 9) # 只有三个点，九个数值都一样才表明是一个三角形
            temp2 = np.zeros(hrwgfunc.shape) # 过滤不合理的取值
            for ind_comp in xrange(3):
                temp2[ind_comp,:,:] = check_matrix*hrwgfunc[ind_comp,:,:]
            G1 = temp2[0]*w2Group_12
            G2 = temp2[1]*w2Group_12
            G3 = temp2[2]*w2Group_12
            G_div = check_matrix*hrwgfunc_div*w2Group_12    
        
           
            return [K12,(F1,F2,F3,F_div),(G1,G2,G3,G_div),(S_matrixinD1,S_matrixinD2)]
        except Exception as e:
            print e
            print self.areas_d.shape
            print triasinD1
            raise
        except AssertionError as ae:
            print ae
            raise
    def fillRHD(self,triasinD1, b_12,rwgs):
        try: 
                       
            hrwginD1_p = [rwg['+'] \
                          for rwg in rwgs[1]]     # 找到所有hrwg
            S_matrix_rowinD1_p = [ii \
                                  for ii,rwg in enumerate(rwgs[1])]
            S_matrix_valuesinD1_p = [1.0 \
                                     for rwg in rwgs[1]]
            hrwginD1_n = [rwg['-'] \
                          for rwg in rwgs[1]]             
            S_matrix_rowinD1_n = [ii \
                                  for ii,rwg in enumerate(rwgs[1])]
            S_matrix_valuesinD1_n = [-1.0 \
                                     for rwg in rwgs[1]]
            
            hrwginD1 = hrwginD1_p + hrwginD1_n
            S_matrix_rowinD1 = S_matrix_rowinD1_p + S_matrix_rowinD1_n
            S_matrix_colinD1 = np.arange(len(hrwginD1))
            S_matrix_valuesinD1 = S_matrix_valuesinD1_p + S_matrix_valuesinD1_n
       
            S_matrix = coo_matrix((S_matrix_valuesinD1,(S_matrix_rowinD1, S_matrix_colinD1)),\
                                      shape=[len(rwgs[1]),len(hrwginD1)])
            
            d1 = self.d
            
            num1 = b_12.numPoint()
            r1Group_12 = self.r_quad["b_%d%d"%(b_12.degree(),b_12.rule())]\
                                     [triasinD1]
            r1Group_12 = r1Group_12.reshape([-1,3])
            w1Group_12 = self.w_quad["b_%d%d"%(b_12.degree(),b_12.rule())]\
                                     [triasinD1]
            w1Group_12 = w1Group_12.reshape([-1])
            r1Group_12_find = [ii for ii,cell in enumerate(d1) \
                               for ind in xrange(num1)]
            r1Group_12_find = np.array(r1Group_12_find) # NofTria
            
            
            #生成F
            Tria_Hrwg = np.array([ff[0] for ff in hrwginD1])             
            FreePoint_Hrwg = np.array([ff[0][ff[1]] for ff in hrwginD1]) # Hrwg*3 -- 所有hrwg的自由节点
            Weight_Hrwg = np.array([ff[2] for ff in hrwginD1]).reshape([1,-1,1]) # Hrwg*3 -- 所有hrwg的权重
                 
            r1Group_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]]) # 3*Hrwg*r -- 积分点
            free_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]])  # 3*Hrwg*r -- 所有hrwg的自由节点
            for id_comp in xrange(3): # 将其进行meshgrid
                r1Group_Hrwg[id_comp,:,:],free_Hrwg[id_comp,:,:] = np.meshgrid(r1Group_12[:,id_comp], FreePoint_Hrwg[:,id_comp])
            hrwgfunc = Weight_Hrwg*(r1Group_Hrwg - free_Hrwg) # 所有hrwg的取值
            
            temp_check_r_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]]) # 检查hrwg的合理值，去掉一些支撑集外面的非零值
            temp_check_hrwg_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r1Group_12.shape[0]]) 
            class temp0(object):
                def __init__(self):
                    pass
                def iter(self, id):
                    id_x,id_y = id
                    temp_check_r_tria[id_x,id_y,:,:], temp_check_hrwg_tria[id_x,id_y,:,:] \
                                 = np.meshgrid(d1[r1Group_12_find][:,id_x,id_y], Tria_Hrwg[:,id_x,id_y])
                    pass
            map(temp0().iter, [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)])
            temp_check = (temp_check_r_tria == temp_check_hrwg_tria) # 判断三角形是否一致
            check_matrix = (np.sum(np.sum(temp_check,axis=0),axis=0) == 9) # 只有三个点，九个数值都一样才表明是一个三角形
            temp2 = np.zeros(hrwgfunc.shape) # 过滤不合理的取值
            for ind_comp in xrange(3):
                temp2[ind_comp,:,:] = check_matrix*hrwgfunc[ind_comp,:,:]
            F1 = temp2[0]*w1Group_12
            F2 = temp2[1]*w1Group_12
            F3 = temp2[2]*w1Group_12

            # 形成K矩阵
            temp = self.planewave(r1Group_12)
            K1 = temp[:,0]
            K2 = temp[:,1]
            K3 = temp[:,2]
            return S_matrix.dot(F1.dot(K1)+ F2.dot(K2)+ F3.dot(K3)).reshape([-1,1])
        except Exception as e:
            print e
            raise
    def fillField(self, r_obs, I_current, triasinD2, b_12, rwgs): 
        try:
            
            # 分区三角形的节点
            d2 = self.d # NofTria _3 _3
            
            # 收集高斯点
            num2 = b_12.numPoint()
            r2Group_12 = self.r_quad["b_%d%d"%(b_12.degree(),b_12.rule())]\
                                     [triasinD2]
            r2Group_12 = r2Group_12.reshape([-1,3])            
            w2Group_12 = self.w_quad["b_%d%d"%(b_12.degree(),b_12.rule())]\
                                     [triasinD2]
            w2Group_12 = w2Group_12.reshape([-1])
            r2Group_12_find = [ii for ii,cell in enumerate(d2) \
                               for ind in xrange(num2)]
            r2Group_12_find = np.array(r2Group_12_find)# NofTria
           
            # 形成K矩阵
            temp_obs = r_obs.reshape([-1,3])
            assert temp_obs.shape[-1] == 3
            r_1 = np.zeros([3,temp_obs.shape[0],r2Group_12.shape[0]])
            r_2 = np.zeros([3,temp_obs.shape[0],r2Group_12.shape[0]])
            for id_comp in xrange(3):
                r_2[id_comp,:,:],r_1[id_comp,:,:] \
                    = np.meshgrid(r2Group_12[:,id_comp], temp_obs[:,id_comp])
                
            r_vec = r_1-r_2
            r = np.sqrt(np.sum(r_vec*r_vec,axis=0))
            ejkr = scipy.exp(-1j*self.k*r)
            C = 1./r**2\
                *(1+1./(1j*self.k*r))
            
            
            # 形成G矩阵
            hrwginD2_p = [rwg['+'] \
                          for rwg in rwgs[1] if rwg['+'][3] in triasinD2]     # 找到所有hrwg
            S_matrix_rowinD2_p = [ii \
                                  for ii,rwg in enumerate(rwgs[1]) if rwg['+'][3] in triasinD2]
            S_matrix_valuesinD2_p = [1.0 \
                                     for rwg in rwgs[1] if rwg['+'][3] in triasinD2]
            hrwginD2_n = [rwg['-'] \
                          for rwg in rwgs[1] if rwg['-'][3] in triasinD2]             
            S_matrix_rowinD2_n = [ii \
                                  for ii,rwg in enumerate(rwgs[1]) if rwg['-'][3] in triasinD2]       
            S_matrix_valuesinD2_n = [-1.0 \
                                     for rwg in rwgs[1] if rwg['-'][3] in triasinD2]
            
            hrwginD2 = hrwginD2_p + hrwginD2_n
            S_matrix_rowinD2 = S_matrix_rowinD2_p + S_matrix_rowinD2_n
            S_matrix_colinD2 = np.arange(len(hrwginD2))
            S_matrix_valuesinD2 = S_matrix_valuesinD2_p + S_matrix_valuesinD2_n
       
            S_matrixinD2 = coo_matrix((S_matrix_valuesinD2,(S_matrix_rowinD2, S_matrix_colinD2)),\
                                      shape=[len(rwgs[1]),len(hrwginD2)])

            Tria_Hrwg = np.array([ff[0] for ff in hrwginD2]) 
            
            FreePoint_Hrwg = np.array([ff[0][ff[1]] \
                                       for ff in hrwginD2]) # Hrwg*3 -- 所有hrwg的自由节点
            Weight_Hrwg = np.array([ff[2] \
                                    for ff in hrwginD2]).reshape([1,-1,1]) # Hrwg*3 -- 所有hrwg的权重
            
            r2Group_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]]) # 3*Hrwg*r -- 积分点
            free_Hrwg = np.zeros([3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]])  # 3*Hrwg*r -- 所有hrwg的自由节点
            for id_comp in xrange(3): # 将其进行meshgrid
                r2Group_Hrwg[id_comp,:,:],free_Hrwg[id_comp,:,:] \
                    = np.meshgrid(r2Group_12[:,id_comp], FreePoint_Hrwg[:,id_comp])
            hrwgfunc = Weight_Hrwg*(r2Group_Hrwg - free_Hrwg) # 所有hrwg的取值

            temp_check_r_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]]) # 检查hrwg的合理值，去掉一些支撑集外面的非零值
            temp_check_hrwg_tria = np.zeros([3,3,FreePoint_Hrwg.shape[0],r2Group_12.shape[0]]) 
            class temp0(object):
                def iter(self,id):
                    id_x,id_y = id
                    temp_check_r_tria[id_x,id_y,:,:], temp_check_hrwg_tria[id_x,id_y,:,:] \
                        = np.meshgrid(d2[r2Group_12_find][:,id_x,id_y], Tria_Hrwg[:,id_x,id_y])
                    pass
            map(temp0().iter,[(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)] )
            temp_check = (temp_check_r_tria == temp_check_hrwg_tria) # 判断三角形是否一致
            check_matrix = (np.sum(np.sum(temp_check,axis=0),axis=0) == 9) # 只有三个点，九个数值都一样才表明是一个三角形
            temp2 = np.zeros(hrwgfunc.shape) # 过滤不合理的取值
            for ind_comp in xrange(3):
                temp2[ind_comp,:,:] = check_matrix*hrwgfunc[ind_comp,:,:]
            G1 = temp2[0]*w2Group_12
            G2 = temp2[1]*w2Group_12
            G3 = temp2[2]*w2Group_12
            
            assert I_current.shape[1] == 1
            X = S_matrixinD2.T.dot(I_current)
            m1 = G1.T.dot(X)
            m2 = G2.T.dot(X)
            m3 = G3.T.dot(X)
            m = np.hstack((m1,m2,m3))
            

            M = np.sum(r_vec.transpose([1,2,0])*m,axis=-1)*r_vec/r**2
            M = M.transpose([1,2,0])
            m = m.reshape([1,-1,3])
            temp1C = 1j*self.k/r+C
            temp1Cx = (M-m).transpose([2,0,1])*temp1C
            temp2C = 2*M.transpose([2,0,1])*C
            result = ((temp1Cx+temp2C)*ejkr).transpose([1,2,0])*self.aita/4/np.pi
            result = np.sum(result, axis=1)

            return result.reshape([r_obs.shape[0],r_obs.shape[1],3])
        except Exception as e:
            print e
            raise
        except AssertionError as ae:
            print ae
            raise       
            
# In[]
class ImpMatrix(object):
    def __init__(self, impMatrix):
        self.impMatrix = impMatrix[0]
        self.coef = impMatrix[-1]
        pass
            
    def matVec(self,vector):
        return self.matVec0(vector)
    
    def rmatVec(self,vector):
        return self.rmatVec0(vector)
    
    def matVec0(self,vector):
        try:
            raise
            pass
        except Exception as e:
            print e
            raise

    def rmatVec0(self,vector):
        try:
            raise
            pass
        except Exception as e:
            print e
            raise
            
class ImpMatrix1(object):
    def __init__(self, impMatrix):
        self.impMatrix = impMatrix[0]
        self.coef = impMatrix[-1]
        self.merge_full()
        pass

    def merge_kernel(self, id_matrix):
        try:
            matrix = self.impMatrix[id_matrix]
            if len(matrix)==3:
                GX1 = matrix[1][0]
                GX2 = matrix[1][1]
                GX3 = matrix[1][2]
                GX4 = matrix[1][3]
                # FKGX = F.dot(KGX)
                FKGX1 = matrix[0][0].dot(GX1)
                FKGX2 = matrix[0][1].dot(GX2)
                FKGX3 = matrix[0][2].dot(GX3)
                FKGX4 = matrix[0][3].dot(GX4)
                pass
            elif len(matrix)==4:
                GX1 = matrix[2][0].T
                GX2 = matrix[2][1].T
                GX3 = matrix[2][2].T
                GX4 = matrix[2][3].T
                # KGX = K.dot(GX)
                KGX1 = matrix[0].dot(GX1)
                KGX2 = matrix[0].dot(GX2)
                KGX3 = matrix[0].dot(GX3)
                KGX4 = matrix[0].dot(GX4)
                # FKGX = F.dot(KGX)
                FKGX1 = matrix[1][0].dot(KGX1)
                FKGX2 = matrix[1][1].dot(KGX2)
                FKGX3 = matrix[1][2].dot(KGX3)
                FKGX4 = matrix[1][3].dot(KGX4)
                pass
            else:
                raise IndexError
            FKGs =  self.coef[0]*(FKGX1+FKGX2+FKGX3)\
                        +self.coef[1]*FKGX4     
            return matrix[-1][0].dot(FKGs.dot(matrix[-1][1].T))
            pass
        except AssertionError as ae:
            print ae
            raise
        except Exception as e:
            print e
            raise
            
    def merge_full(self):
        try:
            sFKGs = map(self.merge_kernel,xrange(len(self.impMatrix)))
            self.full_matrix = np.sum(np.array(sFKGs),axis=0)            
            pass
        except Exception as e:
            print e
            raise
            
    def matVec(self,vector):
        return self.matVec0(vector)
    
    def rmatVec(self,vector):
        return self.rmatVec0(vector)
    
    def matVec0(self,vector):
        try:
            return self.full_matrix.dot(vector.reshape([-1,1]))
            pass
        except Exception as e:
            print e
            raise

    def rmatVec0(self,vector):
        try:
            return self.full_matrix.T.dot(vector.reshape([-1,1]))
            pass
        except Exception as e:
            print e
            raise

def run(cls_instance, var):
    return cls_instance.kernel(var)

class ImpMatrix2(ImpMatrix):
    def __init__(self, impMatrix):
        ImpMatrix.__init__(self,impMatrix)
        self.FKGs = list()
        self.merge()
        pass
    def merge_kernel(self, id_matrix):
        try:
            matrix = self.impMatrix[id_matrix]
            if len(matrix)==3:
                GX1 = matrix[1][0]
                GX2 = matrix[1][1]
                GX3 = matrix[1][2]
                GX4 = matrix[1][3]
                # FKGX = F.dot(KGX)
                FKGX1 = matrix[0][0].dot(GX1)
                FKGX2 = matrix[0][1].dot(GX2)
                FKGX3 = matrix[0][2].dot(GX3)
                FKGX4 = matrix[0][3].dot(GX4)
                pass
            elif len(matrix)==4:
                GX1 = matrix[2][0].T
                GX2 = matrix[2][1].T
                GX3 = matrix[2][2].T
                GX4 = matrix[2][3].T
                # KGX = K.dot(GX)
                KGX1 = matrix[0].dot(GX1)
                KGX2 = matrix[0].dot(GX2)
                KGX3 = matrix[0].dot(GX3)
                KGX4 = matrix[0].dot(GX4)
                # FKGX = F.dot(KGX)
                FKGX1 = matrix[1][0].dot(KGX1)
                FKGX2 = matrix[1][1].dot(KGX2)
                FKGX3 = matrix[1][2].dot(KGX3)
                FKGX4 = matrix[1][3].dot(KGX4)
                pass
            else:
                raise IndexError
            return self.coef[0]*(FKGX1+FKGX2+FKGX3)\
                        +self.coef[1]*FKGX4                 
            pass
        except AssertionError as ae:
            print ae
            raise
        except Exception as e:
            print e
            raise
    def merge(self):
        try: 
            self.FKGs = map(self.merge_kernel,xrange(len(self.impMatrix)))    
            pass
        except Exception as e:
            print e
            raise
    
    def matVec0(self,vector):
        try:            
            # FKGXs = []
            v_copy = vector.reshape([-1,1])
            assert v_copy.shape[-1] == 1
            FKGXs = np.zeros([len(self.impMatrix), vector.shape[0]],dtype=np.complex)
            FKGs = self.FKGs
            # 遍历所有分块
            class temp0(object):
                def kernel(self,var):
                    id_matrix, matrix = var
                    X = matrix[-1][1].T.dot(vector)
                    FKGXs[id_matrix,:] = matrix[-1][0].dot(FKGs[id_matrix].dot(X)).reshape([-1])
            map(temp0().kernel, enumerate(self.impMatrix))
            # 累加所有FKGXs,得到FKGXs_sum
            result = np.sum(FKGXs,axis=0)
            return result 
            pass
        except AssertionError as ae:
            print ae
            raise
        except Exception as e:
            print e
            raise
   
    def rmatVec0(self,vector):
        try:
            # X = S_matrix.T.dot(vector)
            v_copy = vector.reshape([-1,1])
            assert v_copy.shape[-1] == 1
            # GKFXs = []
            GKFXs = np.zeros([len(self.impMatrix), vector.shape[0]],dtype=np.complex)
            FKGs = self.FKGs
            # 遍历所有分块
            class temp0(self):
                def kernel(self,var):
                    id_matrix, matrix = var
                    X = matrix[-1][0].T.dot(v_copy)
                    GKFXs[id_matrix,:] = matrix[-1][1].dot(FKGs[id_matrix].T.dot(X)).reshape([-1])
            map(temp0().kernel, enumerate(self.impMatrix))
            # 累加所有GKFXs,得到GKFXs_sum
            result = np.sum(GKFXs,axis=0)
            # return result 
            return result 
            pass
        except AssertionError as ae:
            print ae
            raise
        except Exception as e:
            print e
            raise
# In[]

class Solver(object): # https://docs.scipy.org/doc/scipy-0.16.0/reference/sparse.linalg.html 
    def __init__(self):
        self.threshold = SolverPar().threshold
        pass
    def luSolve(self,matrix, rhd): # 直接求解
        try:
            
            result = scipy.sparse.linalg.spsolve(matrix, rhd)
            return result
            pass
        except:
            raise
    def cgSolve(self,matrix,rhd): # cg迭代求解
        try:
            result = scipy.sparse.linalg.cg(matrix, rhd, tol=self.threshold)
            return result
            pass
        except:
            raise
    def bicgSolve(self,matrix,rhd): # bicg迭代求解
        try:
            result = scipy.sparse.linalg.bicg(matrix, rhd, tol=self.threshold)
            return result
            pass
        except:
            raise

    def bicgstabSolve(self,matrix,rhd): # bicgstab迭代求解
        try:
            result = scipy.sparse.linalg.bicgstab(matrix, rhd, tol=self.threshold)
            return result
            pass
        except:
            raise
    def gmresSolve(self,matrix,rhd): # gmres迭代求解
        try:
            result = scipy.sparse.linalg.gmres(matrix, rhd, tol=self.threshold)
            return result
            pass
        except:
            raise
    def minresSolve(self,matrix,rhd): # 最小余量法求解
        try:
            result = scipy.sparse.linalg.minres(matrix, rhd, tol=self.threshold)
            return result
            pass
        except:
            raise
    def cgsSolve(self,matrix,rhd): # cgs迭代求解
        try:
            result = scipy.sparse.linalg.cgs(matrix, rhd, tol=self.threshold)
            return result
            pass
        except:
            raise
    
# In[]   
class PGreenFunc(object):
    def __init__(self,k, k0,\
                 h=1,d=1,phi=0,\
                 nmax=1,mmax=1):
        self.k = k
        self.k0 = k0
        self.h = h
        self.d = d
        self.phi = phi
        self.nmax = nmax
        self.mmax = mmax
        pass
    def pgf(self, r1,r2):
        n = np.linspace(-self.nmax,self.nmax,2*self.nmax+1)
        m = np.linspace(-self.mmax,self.mmax,2*self.mmax+1)
                
        ms,ns = np.meshgrid(m,n)
        rho_nm = [ns*self.d*np.cos(self.phi),\
                  np.zeros_like(ms),\
                  ms*self.h+ns*self.d*np.sin(self.phi)]
        rho_nm = np.array(rho_nm)
        rho_nm = rho_nm.transpose([1,2,0])
        
        r = r1.reshape([-1,1,1,1,3])
        rp = r2.reshape([1,-1,1,1,3])
        
        r_rp_nm = r-rp-rho_nm
        R_nm = np.sqrt(np.sum(r_rp_nm*r_rp_nm,axis=-1))
        
        k0 = self.k0.reshape([1,-1])
        a = np.exp(-1.j*self.k*R_nm)/R_nm
              
        b = np.exp( -1.j*\
                     np.sum(k0*rho_nm,axis=-1)\
                  )
        
        c = a*b
        
        return np.sum(\
                     np.sum(c,axis=-1),\
                           axis=-1)
    def _ejkr_r(self, r1,r2):
        r1_copy = r1.reshape([-1,1,3])
        r2_copy = r2.reshape([1,-1,3])
        r = r1_copy - r2_copy
        R = np.sqrt(np.sum(r*r,axis=-1))
        return scipy.exp(-1.j*self.k*R)/R

# In[]        
class FillingProcess_DGF_Free(object):
    def __init__(self):
        pass
    def fillingProcess_kernel(self,var):
        try:        
            ii = var['ii']
            jj = var['jj']
            domains_attached = self.domains_attached
            gridinDomain = self.gridinDomain
            filling = self.filling
            triasinDD = self.triasinDD
            b_101 = self.b_101
            rwgs = self.rwgs
            b_31 = self.b_31
            b_41 = self.b_41
            
            ii_index = domains_attached[domains_attached==ii].index.values[0]
            jj_index = domains_attached[domains_attached==jj].index.values[0]
            if ii_index!=jj_index and gridinDomain[ii_index]&gridinDomain[jj_index]: # 两分区相近
                tempZ = filling.fillblock_dgf_free(triasinDD[ii_index], triasinDD[jj_index], b_101, b_31, rwgs)   
            elif ii_index == jj_index: # 两分区相同
                tempZ = filling.fillblock_dgf_free(triasinDD[ii_index], triasinDD[jj_index], b_101, b_41, rwgs)
            else: # 两分区远离
                tempZ = filling.fillblock_dgf_free(triasinDD[ii_index], triasinDD[jj_index], b_101, b_101, rwgs)
                pass
            
            r1Group_12,r2Group_12 = tempZ[0]
            return tempZ              
            pass
        except Exception as e:
            print e
            raise    
    
    def fillingProcess_dgf_free(self, k,grids,trias,rwgs,domains,gridinDomain,triasinDomain): # 计算所有区域的高斯点
        try:
            # 设置高斯积分
            quadRule = QuadRule()
            b_101 = quadRule.b_101
            b_31 = quadRule.b_31
            b_41 = quadRule.b_41
            filling_method4near_with_sig = quadRule.filling_method4near_with_sig
            # 填充矩阵 
            filling =  FillingMatrix_dgf_free(k,grids,trias)
            triasinDD = triasinDomain
                        
            self.domains_attached = pds.Series(domains)
            self.gridinDomain = gridinDomain
            self.filling_method4near_with_sig = filling_method4near_with_sig
            self.filling = filling
            self.triasinDD = triasinDD
            self.b_101 = b_101
            self.rwgs = rwgs
            self.b_31 = b_31
            self.b_41 = b_41
            self.k = k
            
            domainPairs = list(itertools.product(domains,repeat=2))
            myVars = [{"ii":ii,"jj":jj} for ii,jj in domainPairs]    
                    
            impMatrix = map(self.fillingProcess_kernel, myVars)
            
            # 填充右端激励项
            rhdTerm = [len(rwgs[1]),len(rwgs[1])]
            return [[impMatrix,rhdTerm,filling.coef_equations()],filling]
        except AssertionError as aes:
            print aes
            raise
            
    def fillingRHD_dgf_free(self, trias,rwgs, filling):
        try:
            # 设置高斯积分
            quadRule = QuadRule()
            b_21 = quadRule.b_21
            # 填充右端激励项
            rhdTerm = filling.fillRHD(xrange(len(trias)), b_21, rwgs) 
            return rhdTerm
        except AssertionError as aes:
            print aes
            raise    
    def fillingDGF(self, impMatrix, filling):
        try:
            cellPar = CellPar()
            dgf = PGreenFunc(filling.k, filling.getk_direct(), cellPar.h, cellPar.d, cellPar.phi, cellPar.nmax, cellPar.mmax)
            class temp0(object):
                def iter(self,matrix):
                    r1Group, r2Group = matrix[0]
                    if True == cellPar.isPeriod:
                        return dgf.pgf(r1Group, r2Group)
                    else:
                        return dgf._ejkr_r(r1Group, r2Group)
            return map(temp0().iter, impMatrix)
            pass
    	except Exception as e:
            print e
            raise
# In[]
def getFarFiled(r_obs,  I_current, fillinghander, trias, rwgs):
    try:

        quadRule = QuadRule()
        b_101 = quadRule.b_101

        return fillinghander.fillField(r_obs, I_current, xrange(len(trias)), \
                                       b_101,rwgs)
        pass
    except AssertionError as ae:
        print ae
        raise
    except Exception as e:
        print e
        raise
