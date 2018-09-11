# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 08:58:29 2018

@author: 913
"""

import numpy as np

from _myutils import Cubature

#from _rankRevealing import *

class Filename(object):
    def __init__(self, filename = 'plane'):
        self.filename = filename

class DomainSeg(object):
    def __init__(self, segment =[-1,1,1] ):
        self.segment = segment
        
class SolverPar(object):
    def __init__(self,matrix_solver_type = 'dir_dgf_free'):
        self.matrix_solver_type = matrix_solver_type
        self.aca_threshold = 1.e-4
        self.threshold = 1.e-10

class CellPar(object):
    def __init__(self):
        self.d = 2
        self.h = 1
        self.phi = np.pi*0
        self.nmax = 10
        self.mmax = 10
        self.isPeriod = True
        
class RCSPar(object):
    def __init__(self):
        self.whichPlan = 'phi'
        self.whichPol = 'TE'
        self.numthread = 1
        
class RCSPar_theta(object):
    def __init__(self):
        self.theta_0 = np.linspace(0,180,37)*np.pi/180.
        self.phi_0 = np.array([90,])*np.pi/180.
        self.r = 100.

class RCSPar_phi(object):
    def __init__(self):
        self.theta_0 = np.array([90,])*np.pi/180.
        self.phi_0 = np.linspace(0,180,61)*np.pi/180.
#        self.phi_0 = np.array([90,])*np.pi/180.
        self.r = 100.
        
class WorkingFreqPar(object):
    def __init__(self):
        self.wavenumber=np.pi*2
        
class QuadRule(object):
    def __init__(self):
        self.b_101 = Cubature(1,1) #out-interg
        self.b_21 = Cubature(2,1) #Far
        self.b_31 = Cubature(3,1) #Neighbor
        self.b_41 = Cubature(4,1) #Same
        self.filling_method4near_with_sig = True

class IncidentPar(object):
    def __init__(self):
        self.k_direct = np.array([0,-1,0.]).reshape([1,3])
        self.e_direct = np.array([0.,0,1.]).reshape([1,3])
