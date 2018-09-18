# -*- coding: utf-8 -*-
# common packages
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import LinearOperator
import datetime
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import pandas as pds
import itertools
from multiprocessing.dummy import Pool

# defined packages
from _domain_utils import autoDecomDomain, optDecomDomain,optDecomDomain_check
#import pyximport; pyximport.install()
from Components import FillingProcess_DGF_Free
from Components import ImpMatrix2
from Components import Solver,getFarFiled
from Components import RWGFunc

# In[] Some common parameters
from Parameters import Filename,DomainSeg,SolverPar,RCSPar_theta,RCSPar_phi, WorkingFreqPar, IncidentPar, RCSPar

# In[]
def plotGeo(grids,trias):
    try:
        assert trias.shape[1] >= 3        
        x = grids[:,0]
        y = grids[:,1]
        z = grids[:,2]
        
        if trias.shape[-1] == 4:
            domainID = np.unique(trias[:,3])
            domainID_attached = pds.Series(domainID)
            center_domain = np.zeros([len(domainID),3])
            num_domain = np.zeros([len(domainID),1])
            for tria in trias:
                index_tria = domainID_attached[domainID_attached==tria[3]].index.values[0]
                center_domain[index_tria] = center_domain[index_tria] + \
                    np.array([np.mean(x[tria[:-1]]),np.mean(y[tria[:-1]]),np.mean(z[tria[:-1]])])
                num_domain[index_tria] = num_domain[index_tria]+1
            center_domain  = center_domain/num_domain    
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        colors_key = colors.keys()
        if trias.shape[-1] == 4:
            try:
                for tria in trias:
                    indx = np.hstack((tria[:-1],tria[0]))
                    ax.plot(x[indx], y[indx], z[indx],color=colors_key[0])
            except Exception as e:
                print e
                raise
            for ii in xrange(center_domain.shape[0]):
                ax.text(center_domain[ii,0],center_domain[ii,1],center_domain[ii,2], 'D-%d'%ii,color='black')
        else:
            try:
                for tria in trias:
                    indx = np.hstack((tria,tria[0]))
                    ax.plot(x[indx], y[indx], z[indx],color=colors_key[0])
            except Exception as e:
                print e
                raise
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
        
        pass
    except AssertionError as ae:
        print ae
        raise
    except Exception as e:
        print e
        raise
        
def plotDomain(grids,trias):
    try:
        assert trias.shape[1] >= 3        
        x = grids[:,0]
        y = grids[:,1]
        z = grids[:,2]
        
        if trias.shape[-1] == 4:
            domainID = np.unique(trias[:,3])
            domainID_attached = pds.Series(domainID)
            center_domain = np.zeros([len(domainID),3])
            num_domain = np.zeros([len(domainID),1])
            for tria in trias:
                index_tria = domainID_attached[domainID_attached==tria[3]].index.values[0]
                center_domain[index_tria] = center_domain[index_tria] + \
                    np.array([np.mean(x[tria[:-1]]),np.mean(y[tria[:-1]]),np.mean(z[tria[:-1]])])
                num_domain[index_tria] = num_domain[index_tria]+1
            center_domain  = center_domain/num_domain
        
                    
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
        colors_key = colors.keys()
        if trias.shape[-1] == 4:
            try:
                for tria in trias:
                    ax.plot_trisurf(x[tria[:-1]], y[tria[:-1]], z[tria[:-1]], \
                                    color = colors_key[tria[-1]%len(colors_key)], linewidth=0.3, antialiased=True)
            except Exception as e:
                print e
                raise
        else:
            try:
                for tria in trias:
                    ax.plot_trisurf(x[tria], y[tria], z[tria], color = colors_key[0], linewidth=0.3, antialiased=True)
            except Exception as e:
                print e
                raise                
        
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        plt.show()
        
        pass
    except AssertionError as ae:
        print ae
        raise
    except Exception as e:
        print e
        raise
                
def preCal(wavenumber, grids, trias, edges, segment):      
    try:     
        trias__,domainGrid, domainIDs  = autoDecomDomain(grids, trias, segment)
        
        if len(optDecomDomain_check(8, grids, trias__,domainGrid, domainIDs)) != 0:
            optDecomDomain(8, grids, trias__,domainGrid, domainIDs)
        
        print "===="*30
        trias = trias__
        domains = np.unique(np.array(trias)[:,3])

        # grouping triangles
        triasinDomain = [[id for id,tria in enumerate(trias) if tria[3]==domainNum] for domainNum in domains]
        gridinDomain = [] # recording all nodes of domains, for determining neighborhood
        for domainNum in domains:
            temp = set()
            for tria in trias:
                if tria[3] == domainNum:
                    temp.update(tria[:-1])
            gridinDomain.append(temp)
    except Exception as e:
        print e
        raise           
    try:  
        k = wavenumber # wavenumber
    except Exception as e:
        print e
        raise          
        raise         
    try:
        # qualifying mesh
        edge_leng = [np.linalg.norm(np.array(grids[edge[0]])-np.array(grids[edge[1]])) for edge in edges] 
        max_len = np.max(np.array(edge_leng))
        if k*max_len < scipy.pi*2*0.2: print 'good mesh'
        else: print 'poor mesh' 
    except Exception as e:
        print 'Skip', e,'-->> Edge'
    try:
        ################################################################## 
        # generating HRWG and RWG
        rwgs = RWGFunc().match(grids, trias)
        return [k,grids,trias,rwgs,domains,gridinDomain,triasinDomain]
    except Exception as e:
        print e
        raise
        
# In[]
def loadMem(filenamePar):
    try:
        # load structure from file
        import triangle as triaPy
        temp = triaPy.load('./',filenamePar.filename)
        temp = triaPy.triangulate(temp,'p') # generating triangles from PLSG file
        grids = temp['vertices']
        
        trias = temp['triangles']
        edges = temp['segments']
        # change structure
        grids = np.hstack( (grids[:,0:1], np.zeros([grids.shape[0],1]), grids[:,1:2]) )

        return [grids,trias,edges]
    except Exception as e:
        print e
        raise
# In[]
def run_solve(cls_instance, var):
    return cls_instance.solve(var)

class Period_FSS(object):
    def simulator(self, filename=Filename(),solverPar=SolverPar()):
        sim_start = datetime.datetime.now()
        ID = "%s_%dH_%dM_%dS_%s_%s"%(\
                                         sim_start.date(),\
                                         sim_start.hour,\
                                         sim_start.minute,\
                                         sim_start.second,\
                                         filename.filename,\
                                         solverPar.matrix_solver_type\
                                         )
        details = dict()
        
        print "load mesh"
        try: 
            
            grids, trias, edges = loadMem(filename)
                    
            wavenumber=WorkingFreqPar().wavenumber
            segment = DomainSeg().segment
            
            k,grids,trias,rwgs,domains,gridinDomain,triasinDomain = \
                preCal(wavenumber, grids, trias, edges,segment)   
    
            plotGeo(grids,trias)
        except Exception as e:
            print e
            raise
            pass  
        details['Triangles'] = trias.shape[0]
        details['RWG'] = len(rwgs[1])
        details['freq'] = 3e8*k/np.pi/2
        details['wavenumber'] = k
        details['domains'] = len(domains)
        details['triaInDomain'] = triasinDomain
            
        print "Geo-Info"
        print "Triangles  = %d "%(details['Triangles'])        
        print "RWG Funcs = %d "%(details['RWG'])
        print 'frequency = %.2e Hz'%(details['freq'])
        print "wavenumber = %.2e"%(details['wavenumber'])
        print "domains = %d"%details['domains']
        
        try:
            matrix_solver_type = solverPar.matrix_solver_type
        except Exception as e:
            print e
            raise
        details['Solver_Info'] = matrix_solver_type
        print "Solver_Info = %s"%details['Solver_Info']
                    
        print "filling matrix"
        filling_start = datetime.datetime.now()
        filling_cpu_start = time.clock()
        print 'filling start @ ', filling_start
        try:
            fillingProcess = FillingProcess_DGF_Free()
            matrix_all_in, filling_hander = fillingProcess.fillingProcess_dgf_free(\
                                                                       k,grids,trias,rwgs,domains,gridinDomain,triasinDomain\
                                                                       )
        except Exception as e:
            print e
            raise
        
        filling_end = datetime.datetime.now()
        filling_cpu_end = time.clock()
        details['fillingtime'] = (filling_end-filling_start).seconds
        details['fillingcputime'] = filling_cpu_end-filling_cpu_start
        print 'filling end @ ', filling_end
        print 'filling time = %.2e s = %.2e m' %(details['fillingtime'], details['fillingtime']/60.)
        print "cpu time = %.2e s = %.2e m"%(details['fillingcputime'], details['fillingcputime']/60.)
        print "--"*90
        
        # 定义一个内部函数，用来做循环
        ############################
        class solving_kernel(object):
            def __init__(self,k_dirs__,e_dirs__):
                self.k_dirs = k_dirs__
                self.e_dirs = e_dirs__
                details['rhd'] = dict()
                details['current'] = dict()
                details['f_e']=np.zeros([self.k_dirs.shape[0],self.k_dirs.shape[1]])
                details['f_h']=np.zeros([self.k_dirs.shape[0],self.k_dirs.shape[1]])
                pass
            
            def solve(self, ind_inc_i_j):
                try:
                    print ind_inc_i_j
                    incPar = IncidentPar()
                    incPar.k_direct = self.k_dirs[ind_inc_i_j[0],ind_inc_i_j[1]].reshape([-1,3])# 
                    incPar.e_direct = self.e_dirs[ind_inc_i_j[0],ind_inc_i_j[1]].reshape([-1,3])#
                    filling_hander.changeIncDir(incPar) 
                except Exception as e:
                    print e
                    raise
                try:                    
                    DGF = fillingProcess.fillingDGF(matrix_all_in[0],filling_hander)
                    class temp0(object):
                        def iter(self, id):
                            return [DGF[id],matrix_all_in[0][id][1],matrix_all_in[0][id][2],matrix_all_in[0][id][3]]
                    matrixs = map(temp0().iter, xrange(len(DGF)))
                    matrix_all_in_dgf = [matrixs,matrix_all_in[1],matrix_all_in[2]]
                    matrix = LinearOperator((matrix_all_in_dgf[1][0],matrix_all_in_dgf[1][1]),\
                                                    ImpMatrix2(matrix_all_in_dgf).matVec,ImpMatrix2(matrix_all_in_dgf).rmatVec)
                except Exception as e:
                    print e
                    raise               
                try:    
                    rhdTerm = fillingProcess.fillingRHD_dgf_free(trias,rwgs,filling_hander)
                except Exception as e:
                    print e
                    raise
    
                try:
                    result1 = Solver().cgsSolve(matrix, rhdTerm) 
                    assert  result1[1] == 0
                    I_current = result1[0].reshape([-1,1])
                except Exception as e:
                    print e
                    raise 
                    
                details['rhd']["%d_%d"%(ind_inc_i_j[0],ind_inc_i_j[1])] = rhdTerm
                details['current']["%d_%d"%(ind_inc_i_j[0],ind_inc_i_j[1])] = I_current
                
                try:           
                    tempRCSPar = RCSPar_phi()
                    r = tempRCSPar.r
                    r_obs = -incPar.k_direct*r
                    r_obs = np.array(r_obs).reshape([1,1,-1])
                    field_obs = getFarFiled(r_obs, I_current, filling_hander, trias, rwgs) 
                except Exception as e:
                    print e
                    raise                            
                try:        
                    #1
                    field_e = np.multiply(field_obs,incPar.e_direct)
                    field_e = np.sum(field_e, axis=2)
                    field_e = np.multiply(field_e,np.conj(field_e))
                    aug = np.abs(field_e)*r**2*4*np.pi
                except Exception as e:
                    print e
                    raise              
                details['f_e'][ind_inc_i_j[0],ind_inc_i_j[1]] = aug[0,0]  
                       
                try:        
                    #1
                    field_h = np.multiply(field_obs,np.cross(incPar.k_direct,incPar.e_direct))
                    field_h = np.sum(field_h, axis=2)
                    field_h = np.multiply(field_h,np.conj(field_h))
                    aug = np.abs(field_h)*r**2*4*np.pi
                except Exception as e:
                    print e
                    raise                 
                details['f_h'][ind_inc_i_j[0],ind_inc_i_j[1]]= aug[0,0]
                pass
        ####################################### 
        print "solving equation"
        solving_start = datetime.datetime.now()
        solving_cpu_start = time.clock()
        print 'solving start @ ', solving_start     
        try:
            rCSPar = RCSPar()
            if 'theta'==rCSPar.whichPlan:
                RCS_plane = RCSPar_theta()
            else:
                RCS_plane = RCSPar_phi()
            thetas = RCS_plane.theta_0.reshape([-1,1])#np.array([np.pi/2,]).reshape([-1,1])
            phis = RCS_plane.phi_0.reshape([1,-1]) #np.linspace(np.pi*0,np.pi*1.,3).reshape([1,-1])
            k_dirs = np.array([np.sin(thetas)*np.cos(phis),\
                               np.sin(thetas)*np.sin(phis),\
                               np.cos(thetas)*np.ones_like(phis)])\
                                    .transpose([1,2,0])
            
            v_dirs = np.array([-np.ones_like(thetas)*np.sin(phis),\
                                   np.ones_like(thetas)*np.cos(phis),\
                                   np.zeros([thetas.shape[0],phis.shape[1]])])\
                                        .transpose([1,2,0])
            if "TE" == rCSPar.whichPol:      # TE            
                e_dirs = v_dirs
            else:   # TM
                h_dirs = v_dirs
                e_dirs = np.cross(h_dirs,k_dirs)
    
            solver = solving_kernel(k_dirs,e_dirs)
#            pool = Pool(rCSPar.numthread)
#            for var in list(itertools.product(xrange(thetas.shape[0]),xrange(phis.shape[1]))):
#                pool.apply_async(run_solve, (solver, var))
#            pool.map(solver.solve, list(itertools.product(xrange(thetas.shape[0]),xrange(phis.shape[1]))))
#            pool.close()
#            pool.join()   
            map(solver.solve, list(itertools.product(xrange(thetas.shape[0]),xrange(phis.shape[1]))))
        except Exception as e:
            print e
            raise
            
        solving_end = datetime.datetime.now()
        solving_cpu_end = time.clock()
        details['solveingtime'] = (solving_end-solving_start).seconds
        details['solveingcputime'] = solving_cpu_end-solving_cpu_start
        details['theta'] = thetas
        details['phi'] = phis
        print 'solving end @ ', solving_end
        print 'solving time =  %.2e s = %.2e m'%(details['solveingtime'], details['solveingtime']/60.)
        print "cpu time = %.2e s = %.2e m"%(details['solveingcputime'], details['solveingcputime']/60.)
        print "--"*90
    
        return (ID,details)
        
if __name__ == '__main__':
    import os
    import pickle
    try:
        os.mkdir('result')
    except Exception as e:
        print e
        
    ID_sim,details_sim = Mono_Static_RCS().simulator(filename=Filename('butterfly'),solverPar=SolverPar('aca'))
    with open('result/%s.txt'%ID_sim,'w') as f:
        pickle.dump(details_sim,f)

    

        
