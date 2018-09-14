# -*- coding: utf-8 -*-


# In[] Possion Transform
import scipy as np
import scipy.special as spf
#import pandas as pds

class PGF_Direct(object):
    def __init__(self,k0, a1, a2):
        self.a1 = a1
        self.a2 = a2
        
        self.mmax = 200
        self.nmax = 200
        self.k0 = k0
    def pgf(self,k_dir_ ,r_):
        a1 = self.a1
        a2 = self.a2
        nmax = self.nmax
        mmax = self.mmax
        k0 = self.k0
        
        k_dir = k_dir_
        k_dir = k_dir.reshape(k_dir.shape[0],1,1,1,-1)        

        r = r_
        r = r.reshape([1,r.shape[0], 1,1,-1])     
        
        m = np.linspace(-mmax,mmax,2*mmax+1)
        n = np.linspace(-nmax,nmax,2*nmax+1)

        rho_mn = m.reshape([1,1,-1,1,1])*a1 + n.reshape([1,1,1,-1,1]) *a2
        R_mn = r-rho_mn
        
        temp1 = np.exp(-1j*k0*np.sum(k_dir*rho_mn,axis=-1))
        R = np.sqrt(np.sum(R_mn*R_mn,axis=-1))
        
        temp2 = np.exp(-1j*k0*R)/R/np.pi/4
        
        return np.sum(np.sum(temp1*temp2,axis=-1),axis=-1)
# In[]
#import scipy as np
#import pandas as pds

class PGF_Possion(object):
    def __init__(self,k0, a1, a2):
        self.a1 = a1
        self.a2 = a2
        
        self.mmax = 50
        self.nmax = 50
        self.k0 = k0
    def pgf(self,k_dir_ ,r_):
        a1 = self.a1
        a2 = self.a2
        nmax = self.nmax
        mmax = self.mmax
        k0 = self.k0
        
        k_dir = k_dir_
        k_dir = k_dir.reshape(k_dir.shape[0],1,1,1,-1)
        
        r = r_
        r = r.reshape([1,r.shape[0], 1,1,-1])
        # 计算\hat K_1和\hat K_2
        a1Xa2 = np.cross(a1,a2)
        _Omega = np.sqrt(np.sum((a1Xa2)**2,axis=-1))
        
        _hat_K_1 = np.cross(a2,np.cross(a1,a2))/_Omega**2*np.pi*2.
        _hat_K_2 = np.cross(a1,np.cross(a2,a1))/_Omega**2*np.pi*2.
        _hat_K_3_unit = np.cross(_hat_K_1, _hat_K_2)
        _hat_K_3_unit = _hat_K_3_unit/(np.sqrt(np.sum(_hat_K_3_unit*_hat_K_3_unit)))
        # 计算k_rho
        
        _hat_k_inc = k_dir*k0
        _hat_k_inc = _hat_k_inc/np.sqrt(np.sum(_hat_k_inc*_hat_k_inc))
        k_rho = _hat_k_inc - np.sum(_hat_k_inc*_hat_K_3_unit)*_hat_K_3_unit
        # 计算\hat K_mn
        
        m = np.linspace(-mmax,mmax,2*mmax+1)
        n = np.linspace(-nmax,nmax,2*nmax+1)
        #print m
        #print n
        _hat_K_mn = m.reshape([1,1,-1,1,1])*_hat_K_1.reshape([1,1,1,1,-1])\
            +n.reshape([1,1,1,-1,1])*_hat_K_2.reshape([1,1,1,1,-1])\
            -k_rho.reshape([k_rho.shape[0],1,1,1,-1])
            
        # 计算\gamma_z
        K_mn_2 = np.sum(_hat_K_mn*_hat_K_mn,axis=-1)
        _gamma_z = np.sqrt(K_mn_2-k0**2)
        
        # 计算
        z = r[:,:,:,:,-1]
        z = np.absolute(z)
        #print z
        
        temp1 = np.exp(-_gamma_z*z)/_gamma_z
        temp2 = np.exp(\
                1.j*np.sum(_hat_K_mn*r,axis=-1)\
                )
        temp = temp1*temp2
        result = np.sum(np.sum(temp,axis=-1),axis=-1)/_Omega
        return result

# In[] 
#import scipy as np
class PGF_EWALD(object):
    def __init__(self,k0, a1, a2):
        self.a1 = a1
        self.a2 = a2
        
        self.mmax = 50
        self.nmax = 50
        self.k0 = k0
        pass
    def pgf(self,k_dir_ ,r_):
        a1 = self.a1
        a2 = self.a2
        nmax = self.nmax
        mmax = self.mmax
        k0 = self.k0
        
        k_dir = k_dir_
        k_dir = k_dir.reshape(k_dir.shape[0],1,1,1,-1)
        
        r = r_
        r = r.reshape([1,r.shape[0], 1,1,-1])
        # 计算\hat K_1和\hat K_2
        a1Xa2 = np.cross(a1,a2)
        _Omega = np.sqrt(np.sum((a1Xa2)**2,axis=-1))
        
        _hat_K_1 = np.cross(a2,np.cross(a1,a2))/_Omega**2*np.pi*2.
        _hat_K_2 = np.cross(a1,np.cross(a2,a1))/_Omega**2*np.pi*2.
        _hat_K_3_unit = np.cross(_hat_K_1, _hat_K_2)
        _hat_K_3_unit = _hat_K_3_unit/(np.sqrt(np.sum(_hat_K_3_unit*_hat_K_3_unit)))
        # 计算k_rho
        
        _hat_k_inc = k_dir*k0
        _hat_k_inc = _hat_k_inc/np.sqrt(np.sum(_hat_k_inc*_hat_k_inc))
        k_rho = _hat_k_inc - np.sum(_hat_k_inc*_hat_K_3_unit)*_hat_K_3_unit
        # 计算\hat K_mn
        
        m = np.linspace(-mmax,mmax,2*mmax+1)
        n = np.linspace(-nmax,nmax,2*nmax+1)
        #print m
        #print n
        _hat_K_mn = m.reshape([1,1,-1,1,1])*_hat_K_1.reshape([1,1,1,1,-1])\
            +n.reshape([1,1,1,-1,1])*_hat_K_2.reshape([1,1,1,1,-1])\
            -k_rho.reshape([k_rho.shape[0],1,1,1,-1])
            
        # 计算\gamma_z
        K_mn_2 = np.sum(_hat_K_mn*_hat_K_mn,axis=-1)
        _gamma_z = np.sqrt(K_mn_2-k0**2)
        z = r[:,:,:,:,-1]  
        
        E = np.sqrt(np.pi/_Omega)
        
        temp_pos_1 = np.exp(_gamma_z*z)/_gamma_z*spf.erfc(_gamma_z/2./E+z*E)
        temp_neg_1 = np.exp(-_gamma_z*z)/_gamma_z*spf.erfc(_gamma_z/2./E-z*E)
        
        temp_1 = temp_pos_1+temp_neg_1
        temp2_1 = np.exp(\
                1.j*np.sum(_hat_K_mn*r,axis=-1)\
                )  
        _Psi_1 = np.sum(np.sum(temp_1*temp2_1,axis=-1),axis=-1)
        
        rho_mn = m.reshape([1,1,-1,1,1])*a1 + n.reshape([1,1,1,-1,1]) *a2
        R_mn = r-rho_mn
        R = np.sqrt(np.sum(R_mn*R_mn,axis=-1))
        
        temp_pos_2 = np.exp(1j*k0*R)/R*spf.erfc(R*E+1.j*k0/2/E)     
        temp_neg_2 = np.exp(-1j*k0*R)/R*spf.erfc(R*E-1.j*k0/2/E)
        temp_2 = temp_pos_2+temp_neg_2
        temp2_2 = np.exp(-1.j*np.sum(k_dir*rho_mn,axis=-1))
        _Psi_2 = np.sum(np.sum(temp_2*temp2_2,axis=-1),axis=-1)
        
        result = _Psi_1/4/_Omega +_Psi_2/8/np.pi
        return result
        # 计算
         
        pass
# In[]       
zs = np.linspace(0.0001,0.1,51)
r = np.array([[0,0,zz] for zz in zs]) 
thetas = np.linspace(0,np.pi*0.5,1)
k_dir = np.vstack([np.sin(thetas),np.zeros_like(thetas),np.cos(thetas)])
k_dir = k_dir.transpose()
#k_dir =np.array([[0,1,1],[0,0,1]])
k0 = 2 
a1 = np.array([1,0,0])
a2 = np.array([0,1,0])
        
result_poisson = PGF_Possion(k0,a1,a2).pgf(k_dir,r)
result_direct = PGF_Direct(k0,a1,a2).pgf(k_dir,r)
result_ewald = PGF_EWALD(k0,a1,a2).pgf(k_dir,r)
#print np.absolute(result)
import matplotlib.pylab as plt
plt.figure()
class Method(object):
    def __init__(self,result,marker,line):
        self.result=result
        self.marker=marker
        self.line=line
    def angle(self,it):
        plt.plot(zs,\
                 np.angle(self.result)[it]/np.pi*180,\
                 self.line,\
                 label='angle %d %s'%(it,self.marker))
    def absolute(self,it):
        plt.plot(zs,\
                 np.log10(np.absolute(self.result)[it]),\
                 self.line,\
                 label='abs %d %s'%(it,self.marker))
map(Method(result_poisson,'pois',"-o").angle,xrange(k_dir.shape[0]))
map(Method(result_direct, 'dir',"-").angle,xrange(k_dir.shape[0]))
map(Method(result_ewald, 'ewald',"+").angle,xrange(k_dir.shape[0]))
plt.ylabel("angle (degree)")
plt.xlabel("zs")
plt.legend()
plt.show()

plt.figure()
map(Method(result_poisson,'pois',"-o").absolute,xrange(k_dir.shape[0]))
map(Method(result_direct,'dir',"-").absolute,xrange(k_dir.shape[0]))
map(Method(result_ewald,'ewald',"+").absolute,xrange(k_dir.shape[0]))
plt.xlabel("zs")
plt.ylabel("log10(abs) ")
plt.legend()
plt.show()



plt.figure()
map(Method((result_direct-result_ewald)/result_direct,'diff',"d").absolute,xrange(k_dir.shape[0]))
plt.xlabel("zs")
plt.ylabel("log10(abs) ")
plt.legend()
plt.show()        
        