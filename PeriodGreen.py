# -*- coding: utf-8 -*-


# In[] Poisson Transform
import scipy as np
import scipy.special as spf
import pandas as pds
import warnings

class PGF_Direct(object):
    def __init__(self,k0, a1, a2, nmax=200, mmax=200):
        self.a1 = a1
        self.a2 = a2
        
        self.mmax = mmax
        self.nmax = nmax
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
        
        ejkrho = np.exp(-1j*k0*np.sum(k_dir*rho_mn,axis=-1))
        R = np.sqrt(np.sum(R_mn*R_mn,axis=-1))
#        print R
        
        
        G0 = np.exp(-1j*k0*R)/R/np.pi/4
        
        result = np.sum(np.sum(ejkrho*G0,axis=-1),axis=-1)
#        print result
        
        return result
# In[]
#import scipy as np
#import pandas as pds

class PGF_Poisson(object):
    def __init__(self,k0, a1, a2, nmax=50, mmax=50):
        self.a1 = a1
        self.a2 = a2
        
        self.mmax = mmax
        self.nmax = nmax
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
        k_rho = _hat_k_inc - np.sum(_hat_k_inc*_hat_K_3_unit,axis=-1).reshape([-1,1,1,1,1])*_hat_K_3_unit
        # 计算\hat K_mn
        
        m = np.linspace(-mmax,mmax,2*mmax+1)
        n = np.linspace(-nmax,nmax,2*nmax+1)
        _hat_K_mn = m.reshape([1,1,-1,1,1])*_hat_K_1.reshape([1,1,1,1,-1])\
            +n.reshape([1,1,1,-1,1])*_hat_K_2.reshape([1,1,1,1,-1])\
            -k_rho.reshape([k_rho.shape[0],1,1,1,-1])
            
        # 计算\gamma_z
        K_mn_2 = np.sum(_hat_K_mn*_hat_K_mn,axis=-1)
        _gamma_z = np.sqrt(K_mn_2-k0**2)
    
        # 计算
        z = r[:,:,:,:,-1]
        z = np.absolute(z)
        
        G0_spec = np.exp(-_gamma_z*z)/_gamma_z/2
        ejkR = np.exp(\
                1.j*np.sum(_hat_K_mn*r,axis=-1)\
                )
        temp = G0_spec*ejkR

        result = np.sum(np.sum(temp,axis=-1),axis=-1)/_Omega
        return result

# In[] 
#import scipy as np
class PGF_EWALD(object):
    def __init__(self,k0, a1, a2, nmax=50, mmax=50):
        self.a1 = a1
        self.a2 = a2
        
        self.mmax = mmax
        self.nmax = nmax
        self.k0 = k0
        pass
    def pgf(self,k_dir_ ,r_):## 目前的程序还存在一些问题，发现计算结果和直接计算结果不一致。
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
        k_rho = _hat_k_inc - np.sum(_hat_k_inc*_hat_K_3_unit,axis=-1).reshape([-1,1,1,1,1])*_hat_K_3_unit
        # 计算\hat K_mn
        
        m = np.linspace(-mmax,mmax,2*mmax+1)
        n = np.linspace(-nmax,nmax,2*nmax+1)

        _hat_K_mn = m.reshape([1,1,-1,1,1])*_hat_K_1.reshape([1,1,1,1,-1])\
            +n.reshape([1,1,1,-1,1])*_hat_K_2.reshape([1,1,1,1,-1])\
            -k_rho.reshape([k_rho.shape[0],1,1,1,-1])
            
        # 计算\gamma_z
        K_mn_2 = np.sum(_hat_K_mn*_hat_K_mn,axis=-1)
        _gamma_z = np.sqrt(K_mn_2-k0**2)
        
        z = r[:,:,:,:,-1]  
        E = np.sqrt(np.pi/_Omega)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            term_0_1 = np.where(np.real(_gamma_z)*z<10, np.exp(_gamma_z*z)/_gamma_z, np.ones_like(_gamma_z))
            temp_pos_1 = term_0_1*spf.erfc(_gamma_z/2./E+z*E)
            term_0_2 = np.where(-np.real(_gamma_z)*z<10,  np.exp(-_gamma_z*z)/_gamma_z, np.ones_like(_gamma_z))
            temp_neg_1 = term_0_2*spf.erfc(_gamma_z/2./E-z*E)
        
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
#            raise
        temp_2 = temp_pos_2+temp_neg_2
        temp2_2 = np.exp(-1.j*np.sum(k_dir*rho_mn,axis=-1))
        _Psi_2 = np.sum(np.sum(temp_2*temp2_2,axis=-1),axis=-1)
        
        result = _Psi_1/4/_Omega +_Psi_2/8/np.pi
        return result

# In[]
from scipy.interpolate import RegularGridInterpolator,LinearNDInterpolator
class DGF_Interp_3D(object):
    def __init__(self,x,y,z,\
                 pgf_gen,\
                 k_dir_theta,k_dir_phi): # 插值的角度格子点
        self.x = x
        self.y = y
        self.z = z 
        
        self.pgf_gen = pgf_gen
        self.k_dir_theta = k_dir_theta
        self.k_dir_phi = k_dir_phi
        self.k = pgf_gen.k0
        self.build()
    
    def build(self):
        try:
            x = self.x
            y = self.y
            z = self.z
            theta = self.k_dir_theta
            phi = self.k_dir_phi
            k = self.k
          
            _hat_x = np.array([1,0,0])
            _hat_y = np.array([0,1,0])
            _hat_z = np.array([0,0,1])            
            r = x.reshape([-1,1,1,1])*_hat_x\
                +y.reshape([1,-1,1,1])*_hat_y\
                +z.reshape([1,1,-1,1])*_hat_z
            r_shape_orign = r.shape            
            r_flat = r.reshape([-1,3]) # 几何格子--3D
            
            _hat_kx = np.array([1,0,0])
            _hat_ky = np.array([0,1,0])
            _hat_kz = np.array([0,0,1])
            k_dir = np.sin(theta.reshape([-1,1,1]))*np.cos(phi.reshape([1,-1,1]))*_hat_kx\
                    +np.sin(theta.reshape([-1,1,1]))*np.sin(phi.reshape([1,-1,1]))*_hat_ky\
                    +np.cos(theta.reshape([-1,1,1]))*np.ones_like(phi.reshape([1,-1,1]))*_hat_kz
            k_dir_shape_orign = k_dir.shape 
#            print k_dir
            k_dir_flat = k_dir.reshape([-1,3]) #角度格子--2D
            
            R2_row = np.sum(r_flat*r_flat,axis=-1).reshape([-1]) 
            r_flat = np.array([ r_flat[ii] \
                               if R2_row[ii]>0 else np.array([0,0,1.e-5]) \
                               for ii in xrange(r_flat.shape[0])])# 去掉奇异性
    
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('always')
                    self.data = self.pgf_gen.pgf(k_dir_flat,r_flat)
                    self.data = self.data.reshape(*(k_dir_shape_orign[:-1]+r_shape_orign[:-1]))
            except Exception as e:
                print e
                raise
            
            data_slim_shape = [xx for xx in self.data.shape if xx>1]
            cord_ = (theta,phi,x, y, z)
            cord_slim = [cord_[i] for i in xrange(len(self.data.shape)) if self.data.shape[i]>1]
            self.my_interpolating_function = RegularGridInterpolator(cord_slim, self.data.reshape(data_slim_shape))
        except Exception as e:
            print e
            raise
            
    def interp(self, pts_dir_r):
        try:
            ptr = np.array([pts_dir_r[:,i] for i in xrange(len(self.data.shape)) if self.data.shape[i]>1])
            result = self.my_interpolating_function(ptr.transpose())
            return result
        except Exception as e:
            print e
            raise
        pass    
    
    def interp_dir_r(self, theta_phi, r):
        try:
            pts_dir_r = np.array([np.hstack([xx,yy]) for xx in theta_phi for yy in r])
            result = self.interp(pts_dir_r)
            return result.reshape([theta_phi.shape[0],r.shape[0]])
        except Exception as e:
            print e
            print pts_dir_r.shape
            raise
        pass
    

# In[]  
def test1():
    zs = np.linspace(0.01,1,21)
    r = np.array([[0,0,zz] for zz in zs]) 
    thetas = np.linspace(0,np.pi*0.25,10) 
    print "thetas: ", thetas
    k_dir = np.vstack([np.sin(thetas),np.zeros_like(thetas),np.cos(thetas)])
    k_dir = k_dir.transpose()
    
    wavelength = 10
    k0 = np.pi*2/wavelength
    
    a1 = np.array([1,0,0])
    a2 = np.array([0,1,0])
    class gratinglobes(object): 
        def check(self):
            dx = np.sqrt(np.sum(a1*a1,axis=-1))
            dy = np.sqrt(np.sum(a2*a2,axis=-1))
            threshold_d = wavelength/(1+np.sin(thetas))
            temp = np.array([dx <threshold_d, dy<threshold_d])
            return np.sum(temp,axis=0)==temp.shape[0]
            
    checker =  gratinglobes().check()
    print "grating lobe condition: ", checker

    result_direct = PGF_Direct(k0,a1,a2,100,100).pgf(k_dir,r)     
    result_poisson = PGF_Poisson(k0,a1,a2,1,1).pgf(k_dir,r)
    result_ewald = PGF_EWALD(k0,a1,a2,20,20).pgf(k_dir,r)
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
                     np.absolute(self.result)[it],\
                     self.line,\
                     label='abs %d %s'%(it,self.marker))
           
    map(Method(result_poisson,'pois',"s").angle,xrange(k_dir.shape[0]))
    map(Method(result_direct, 'dir',"-").angle,xrange(k_dir.shape[0]))
    map(Method(result_ewald, 'ewald',"+").angle,xrange(k_dir.shape[0]))
    plt.ylabel("angle (degree)")
    plt.xlabel("zs")
    #plt.legend()
    plt.show()
    
    plt.figure()
    map(Method(result_poisson,'pois',"s").absolute,xrange(k_dir.shape[0]))
    map(Method(result_direct,'dir',"-").absolute,xrange(k_dir.shape[0]))
    map(Method(result_ewald,'ewald',"+").absolute,xrange(k_dir.shape[0]))
    plt.xlabel("zs")
    plt.ylabel("log10(abs) ")
    #plt.legend()
    plt.show()

    plt.figure()
    #map(Method((result_direct-result_poisson)/result_direct,'dir_pois',"-s").absolute,xrange(k_dir.shape[0]))
    map(Method((result_direct-result_ewald)/result_direct,'dir_ewald',"-d").absolute,xrange(k_dir.shape[0]))
    #map(Method((result_poisson-result_ewald)/result_ewald,'pois_ewald',"-o").absolute,xrange(k_dir.shape[0]))
    plt.xlabel("zs")
    plt.ylabel("log10(abs) ")
    #plt.legend()
    plt.show()        
            
    #print result_direct
    #print result_poisson
    #print np.absolute(result_poisson/result_direct)    
#import scipy as np
if __name__ == '__main__':
    '''
    test1()
    '''
    thetas = np.linspace(0,np.pi*0.25,5)
#    thetas = np.array([np.pi*0.3])
    print "thetas: ", thetas
    zmin=1
    zmax=11
    zs = np.linspace(zmin,zmax,200)
    r = np.array([[0,0,zz] for zz in zs]) 

    k_dir = np.vstack([np.sin(thetas),np.zeros_like(thetas),np.cos(thetas)])
    k_dir = k_dir.transpose()
    print k_dir

    wavelength = 10
    k0 = np.pi*2/wavelength
    
    a1 = np.array([1,0,0])
    a2 = np.array([0,1,0])
    class gratinglobes(object): 
        def check(self):
            dx = np.sqrt(np.sum(a1*a1,axis=-1))
            dy = np.sqrt(np.sum(a2*a2,axis=-1))
            threshold_d = wavelength/(1+np.sin(thetas))
            temp = np.array([dx <threshold_d, dy<threshold_d])
            return np.sum(temp,axis=0)==temp.shape[0]
            
    checker =  gratinglobes().check()
    print "grating lobe condition: ", checker
    pgf_gen_ewald = PGF_EWALD(k0,a1,a2,1,1)
    
    pdf =  DGF_Interp_3D(x=np.linspace(0,0.9,1),y=np.linspace(0,0.9,1),z=np.linspace(zmin,zmax,50), \
                         pgf_gen=pgf_gen_ewald,\
                         k_dir_theta=thetas, k_dir_phi=np.array([0,]))

    theta_phi = np.vstack([thetas, np.zeros_like(thetas)])
    theta_phi = theta_phi.transpose()
    result_interp = pdf.interp_dir_r( theta_phi,r)
    result_ewald = pgf_gen_ewald.pgf(k_dir,r)
    
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
                     np.absolute(self.result)[it],\
                     self.line,\
                     label='abs %d %s'%(it,self.marker))
    map(Method(result_interp,'interp',"s").angle,xrange(theta_phi.shape[0]))
    map(Method(result_ewald,'ewald',"-").angle,xrange(theta_phi.shape[0]))
    plt.ylabel("angle (degree)")
    plt.xlabel("zs")
    plt.legend()
    plt.show()
    
    plt.figure()
    map(Method(result_interp,'interp',"+").absolute,xrange(theta_phi.shape[0]))
    map(Method(result_ewald,'ewald',"-").absolute,xrange(theta_phi.shape[0]))
    plt.xlabel("zs")
    plt.ylabel("log10(abs) ")
    plt.legend()
    plt.show()
