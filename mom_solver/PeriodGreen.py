# -*- coding: utf-8 -*-


# In[] Poisson Transform
import scipy as np
import scipy.special as spf
import warnings
import scipy.constants

# 下面是Scala格林函数
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
from scipy.interpolate import RegularGridInterpolator
class DGF_Interp_3D(object):
    def __init__(self,x,y,z,\
                 pgf_gen,\
                 k_dir_theta,k_dir_phi): # 插值的角度格子点
        self.x_sample = x
        self.y_sample = y
        self.z_sample = z 
        
        self.pgf_gen = pgf_gen
        self.k_dir_theta = k_dir_theta
        self.k_dir_phi = k_dir_phi
        self.k = pgf_gen.k0
        self.build()
    
    def build(self):
        try:
            x = self.x_sample
            y = self.y_sample
            z = self.z_sample
            theta = self.k_dir_theta
            phi = self.k_dir_phi
          
            _hat_x = self.pgf_gen.a1
            _hat_y = self.pgf_gen.a2
            _hat_z = np.cross(_hat_x, _hat_y)
            _hat_z = _hat_z/np.sqrt(np.sum(_hat_z*_hat_z))
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
                    R_row = np.sqrt(R2_row)
                    self.data = self.pgf_gen.pgf(k_dir_flat,r_flat)*R_row*np.exp(1j*self.k*R_row)
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
        except ValueError as ve:
            print ve
            print ptr.shape
            print ptr
            raise
        except IndexError as ie:
            print ie
            print i
            print pts_dir_r.shape
            raise
        pass    
    
    def interp_dir_r(self, theta_phi, r):
        try:
            R = np.sqrt(np.sum(r*r,axis=-1))
            R = R.reshape([1,-1])
            
            _hat_x = self.pgf_gen.a1.reshape([1,-1])
            _hat_y = self.pgf_gen.a2.reshape([1,-1])
            _hat_z = np.cross(_hat_x, _hat_y)
            _hat_z = _hat_z/np.sqrt(np.sum(_hat_z*_hat_z))
            
            rx = np.sum(r*_hat_x,axis=-1)/np.sum(_hat_x*_hat_x)
            ry = np.sum(r*_hat_y,axis=-1)/np.sum(_hat_y*_hat_y)
            rz = np.sum(r*_hat_z,axis=-1)
            
            r_loc = np.vstack([rx,ry,rz])
            r_loc = r_loc.transpose()
            theta_phi_copy = theta_phi.reshape([-1,2])
           
            pts_dir_r = np.array([np.hstack([xx,yy]) for xx in theta_phi_copy for yy in r_loc])
            result = self.interp(pts_dir_r)
            return result.reshape([theta_phi.shape[0],r.shape[0]])/R*np.exp(-1j*self.k*R)
        except ValueError as ve:
            print ve
            raise
        except IndexError as ie:
            print ie
            print pts_dir_r.shape
            print r_loc.shape
            print theta_phi.shape
            raise
        pass
    
# In[] 下面是并矢格林函数
class DPGF(object):
    def __init__(self,k0, a1, a2):
        self.a1 = a1
        self.a2 = a2
        
        self.k0 = k0
    def modes(self,k_dir_ ,rho_, m, n):
        try:
            a1 = self.a1
            a2 = self.a2
            k0 = self.k0
            mu0 = np.constants.mu_0
            eps0 = np.constants.epsilon_0
            v0 = np.constants.c
            _circle_omega = k0*v0 
        except Exception as e:
            print e
            raise
        try:
            # 计算\hat K_1和\hat K_2
            a1Xa2 = np.cross(a1,a2)
            _Omega = np.sqrt(np.sum((a1Xa2)**2,axis=-1))
            
            _hat_K_1 = np.cross(a2,np.cross(a1,a2))/_Omega**2*np.pi*2.
            _hat_K_2 = np.cross(a1,np.cross(a2,a1))/_Omega**2*np.pi*2.
            _hat_K_3_unit = np.cross(_hat_K_1, _hat_K_2)
            _hat_K_3_unit = _hat_K_3_unit/(np.sqrt(np.sum(_hat_K_3_unit*_hat_K_3_unit)))
            
            k_dir = k_dir_
            k_dir = k_dir.reshape(k_dir.shape[0],1,1,1,-1)
            
            rho = rho_
            rho = rho.reshape([1,rho.shape[0], 1,1,-1])
            _hat_z = np.array([0,0,1]).reshape([1,1,1,1,-1])
                      
            # 计算k_rho
            _hat_k_inc = k_dir*k0
            k_rho = _hat_k_inc - np.sum(_hat_k_inc*_hat_K_3_unit,axis=-1).reshape([-1,1,1,1,1])*_hat_K_3_unit
            # 计算\hat K_mn       
            _hat_K_mn = m.reshape([1,1,-1,1,1])*_hat_K_1.reshape([1,1,1,1,-1])\
                +n.reshape([1,1,1,-1,1])*_hat_K_2.reshape([1,1,1,1,-1])\
                -k_rho.reshape([k_rho.shape[0],1,1,1,-1]) 
            # 计算\gamma_z
            K_mn_2 = np.sum(_hat_K_mn*_hat_K_mn,axis=-1)
            K_mn = np.sqrt(K_mn_2)
            _gamma_z = np.sqrt(K_mn_2-k0**2)   
    
            _ksi_mn = np.exp(1.j*np.sum(_hat_K_mn*rho,axis=-1))/np.sqrt(_Omega) 
            
            _gamma_z = _gamma_z.reshape( *( _gamma_z.shape+(1,)  ) )
            K_mn = K_mn.reshape( *(K_mn.shape+(1,)))
            _ksi_mn = _ksi_mn.reshape( *(_ksi_mn.shape+(1,)) )
        except Exception as e:
            print e
            raise
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                K_mn_no_zero = np.where(K_mn>0,K_mn,np.ones_like(K_mn))
                _hat_K_mn_unit = _hat_K_mn/K_mn_no_zero
        except Exception as e:
            print e
            raise
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                
                Z_te_mn = 1.j*_circle_omega*mu0/_gamma_z
                _hat_e_te_t_mn = 1.j*_ksi_mn*np.cross(_hat_z,_hat_K_mn_unit)
                _hat_h_te_t_mn = -1.j*_ksi_mn*_hat_K_mn_unit
                _hat_h_te_z_mn = K_mn/_gamma_z*_ksi_mn*_hat_z
                
                Z_tm_mn = _gamma_z/(1.j*_circle_omega*eps0)
                _hat_e_tm_t_mn = _hat_h_te_t_mn
                _hat_e_tm_z_mn = _hat_h_te_z_mn
                _hat_h_tm_t_mn = -_hat_e_te_t_mn
                
                result_mod = [\
                              np.array([_hat_e_te_t_mn, \
                                        _hat_h_te_t_mn+_hat_h_te_z_mn]),\
                              np.array([_hat_e_tm_t_mn+_hat_e_tm_z_mn, \
                                        _hat_h_tm_t_mn])\
                              ]
#                result_imp = np.array([Z_te_mn.reshape([Z_te_mn.shape[0],Z_te_mn.shape[2],Z_te_mn.shape[3]]),\
#                                       Z_tm_mn.reshape([Z_tm_mn.shape[0],Z_tm_mn.shape[2],Z_tm_mn.shape[3]])\
#                                       ])
#                result_gamma = _gamma_z.reshape([_gamma_z.shape[0],_gamma_z.shape[2],_gamma_z.shape[3]])
                result_imp = np.array([Z_te_mn,\
                                       Z_tm_mn\
                                       ])
                result_gamma = _gamma_z
            return [result_mod, result_imp, result_gamma]
        except Exception as e:
            print e
            raise
    def specQuantity(self,z,imp,gamma):
        try:
            return np.exp(\
                          -gamma.reshape( *(gamma.shape+(1,)) )*\
                          np.absolute(z).reshape([-1])\
                        )*imp.reshape(*(imp.shape+(1,)))/2.
            pass
        except Exception as e:
            print e
            raise
    
            
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
def test2():
    thetas = np.linspace(0,np.pi*0.3,7)
#    thetas = np.array([np.pi*0.3])
    print "thetas: ", thetas
    zmin=1
    zmax=5
    zs = np.linspace(zmin,zmax,100)
    r = np.array([[0.1,0.12,zz] for zz in zs]) 

    k_dir = np.vstack([np.sin(thetas),np.zeros_like(thetas),np.cos(thetas)])
    k_dir = k_dir.transpose()
#    print k_dir

    wavelength = 10
    k0 = np.pi*2/wavelength
    
    a1 = np.array([1,0,0])
    a2 = np.array([0,1,0])
    class Gratinglobes(object): 
        def check(self):
            dx = np.sqrt(np.sum(a1*a1,axis=-1))
            dy = np.sqrt(np.sum(a2*a2,axis=-1))
            threshold_d = wavelength/(1+np.sin(thetas))
            temp = np.array([dx <threshold_d, dy<threshold_d])
            return np.sum(temp,axis=0)==temp.shape[0]
            
    checker = Gratinglobes().check()
    print "grating lobe condition: ", checker
    pgf_gen_ewald = PGF_EWALD(k0,a1,a2,2,2)

    theta_phi = np.vstack([thetas, np.zeros_like(thetas)])
    theta_phi = theta_phi.transpose()    
    class DGF_INT(object):
        def interp(self):
            x_sample = np.linspace(0,0.2,3)
            y_sample = np.linspace(0,0.2,3)
            z_sample = np.linspace(zmin,zmax,20)
            theta_sample = np.linspace(0,np.pi*0.67,10)
            phi_sample = np.array([-0.01,0.01])#np.linspace(0,np.pi*2,10)
            pdf =  DGF_Interp_3D(x=x_sample,y=y_sample,z=z_sample, \
                                 pgf_gen=pgf_gen_ewald,\
                                 k_dir_theta=theta_sample, k_dir_phi=phi_sample)
        

            result_interp = pdf.interp_dir_r( theta_phi,r)
            return result_interp
        
    result_interp = DGF_INT().interp()
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

    plt.figure()
    map(Method((result_interp-result_ewald)/result_ewald,'diff',"-+").absolute,xrange(theta_phi.shape[0]))
#    map(Method(result_ewald,'ewald',"-").absolute,xrange(theta_phi.shape[0]))
    plt.xlabel("zs")
    plt.ylabel("log10(abs) ")
    plt.legend()
    plt.show()
#import scipy as np
# In[]
if __name__ == '__main__':
    '''
    test1()
    '''
    '''
    test2()
    '''
    thetas = np.linspace(0,np.pi*0.3,7)
    print "thetas: ", thetas
    rho = np.array([[0.1,0.12,0],[0,0,0]]) 
    print "rho:",rho

    k_dir = np.vstack([np.sin(thetas),np.zeros_like(thetas),np.cos(thetas)])
    k_dir = k_dir.transpose()

    wavelength = 10
    k0 = np.pi*2/wavelength
    
    a1 = np.array([1,0,0])
    a2 = np.array([0,1,0])
    class Gratinglobes(object): 
        def check(self):
            dx = np.sqrt(np.sum(a1*a1,axis=-1))
            dy = np.sqrt(np.sum(a2*a2,axis=-1))
            threshold_d = wavelength/(1+np.sin(thetas))
            temp = np.array([dx <threshold_d, dy<threshold_d])
            return np.sum(temp,axis=0)==temp.shape[0]    
    checker = Gratinglobes().check()
    print "grating lobe condition: ", checker
    dpgf = DPGF(k0,a1,a2)
    modes,imp,gammaz = dpgf.modes(k_dir,rho,np.arange(-2,3),np.arange(-2,3))
    print "TE.modes.shape:", modes[0].shape
    print "TM.modes.shape:", modes[1].shape
    print "imp.shape:", imp.shape
    print "gammaz.shape:", gammaz.shape
    spec =  dpgf.specQuantity(np.array([0.1,0.2]),imp,gammaz)
    print "spec.shape:", spec.shape
