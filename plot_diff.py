# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:31:02 2018

@author: 913
"""
# In[]
import os
try:
    os.mkdir('result')
except Exception as e:
    print e
try: 
    os.mkdir('tempData')
except Exception as e:
    print e
 
# In[]

import pickle
#from multiprocessing import Pool as ProgPool

from mom_solver import Solution
from mom_solver import Parameters

name = "plane"

ID_sim_dir,details_sim_dir = Solution.Period_FSS().simulator(filename=Parameters.Filename(name),\
                                                solverPar=Parameters.SolverPar('dir_dgf_free'))
with open('result/%s.txt'%ID_sim_dir,'w') as f:
    print ID_sim_dir
    pickle.dump(details_sim_dir,f)

# In[]

import matplotlib.pylab as plt
import numpy as np

rCSPar = Parameters.RCSPar()
if 'theta'==rCSPar.whichPlan:
    ## theta-plan
    plt.figure()
    plt.plot(details_sim_dir['theta'][:,0],10*np.log10(\
             details_sim_dir['f'][:,0]))
    plt.legend(fontsize=14)
    plt.xlabel('$\theta$ degree',fontsize=14)
    plt.ylabel('RCS dBsm',fontsize=14)
    xlabel = np.linspace(0,np.pi,7)
    plt.xticks(xlabel,\
               np.array(xlabel*180/np.pi,dtype=float), \
               fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()
    
else:
    # phi-plan
    plt.figure()
    plt.plot(details_sim_dir['phi'][0,:],10*np.log10(\
             details_sim_dir['f'][0,:]))
    plt.legend(fontsize=14)
    plt.xlabel('$\phi$ degree',fontsize=14)
    plt.ylabel('RCS dBsm',fontsize=14)
    xlabel = np.linspace(0,np.pi,7)
    plt.xticks(xlabel,\
               np.array(xlabel*180/np.pi,dtype=float), \
               fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid()
    plt.show()
    pass

# In[] load Feko
'''    
with open('feko.dat') as f:
    lines = f.readlines()

import re
a = re.compile("(.+)\t(.+)\t\n")
b = a.match(lines[3])
def find_data(line):
    try:
        b = a.match(line)
        return [float(b.group(1)), float(b.group(2))]
    except Exception as e:
        print e
        
data_feko = np.array(map(find_data,lines[3:])) 

plt.plot(data_feko[:,0],data_feko[:,1],label="feko")
plt.plot(details_sim_dir['phi'][0,:]*180/np.pi,\
             10*np.log10(details_sim_dir['f_e'][0,:]),label = 'mom_test')
plt.ylim(ymin=-40)
plt.ylim(ymax=20)
plt.legend()
plt.grid()
plt.xlabel("$\phi$")
plt.ylabel("RCS")
plt.show()
'''

