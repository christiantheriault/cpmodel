# %% CREATE FOLDERS
from importlib import reload


import os
import numpy as np
import copy
from shutil import copyfile
os.chdir('/home/dista/Documents')
import sys
sys.path.append('/home/dista/Documents/cpmodel/')

import loaddata
reload(loaddata)
from loaddata import generate_disjunctive_2cat
from loaddata import generate_disjunctive_cat


#dataset='disjunctivevec'
dataset='disjunctivevec'


IMAGEPATH = './datasets/'+dataset+'/images/'
INFOPATH = './datasets/'+dataset+'/info/'

codename1 = 'CP_auto_catego.py'
codename2 = 'loaddata.py'
codename3 = 'net_classes_and_functions.py'
#codename4 = dataset +'_create_folders.py'


codelocation='./cpmodel/'
CODE_PATH='./datasets/'+dataset+'/codes/'
copyfile(codelocation + codename1, CODE_PATH+ codename1 )
copyfile(codelocation + codename2, CODE_PATH+ codename2 ) 
copyfile(codelocation + codename3, CODE_PATH+ codename3 ) 
#copyfile(codelocation + codename4, CODE_PATH+ codename4 ) 

nb=500
N=5
NB_samecondi_runs=1
nb2=1


for  k in range(1,N+1):
    
    if k==1:
            pstart=1
    else:
            pstart=1
    
    for p in range(1,k+1):
        
         for run in range(NB_samecondi_runs):
             
             
            extension = '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)
            # create folders
            catpath = IMAGEPATH + 'images'  + extension + '/'
            cat0 = catpath + 'cat0' + '/'
            cat1 = catpath + 'cat1' + '/'

            if not os.path.exists(cat0):
                os.makedirs(cat0)
                os.makedirs(cat1)
                
            infopath = INFOPATH + 'info'  + extension + '/'
            if not os.path.exists(infopath):
                os.makedirs(infopath)

            A,B,U,covariant_pos = generate_disjunctive_2cat(nb, N , k ,p)
#            cats,U,covariant_pos = generate_disjunctive_cat(nb, N , k ,p)
#            A,B=cats

            A=np.unique(A,axis=0)
            B=np.unique(B,axis=0)


#            nb2=int(3**7/(3**(N-k)))
            nb2=int(np.floor((3**(N-1))/A.shape[0]))

            A=np.tile(A, (nb2, 1))
            B=np.tile(B, (nb2, 1))

            print(A.shape)
            print(B.shape)
            np.save(cat0 + 'A' , A)
            np.save(cat1 + 'B' , B)
            
             
            np.save(infopath + 'U'  , U)
            np.save(infopath + 'covariant_pos' , covariant_pos)

            file = open(infopath + 'info' +  extension + '.txt', 'w')

            file.write('N : ' + str(N) \
                       + os.linesep + 'K: ' + str(k) \
                       + os.linesep + 'P: ' + str(p) + os.linesep \
                       + os.linesep + 'Covariant position(s): ' + ''.join(str(covariant_pos)) + os.linesep \
                       + os.linesep + 'Dimension assigned value(s): ' + os.linesep + ''.join(str(U)) + os.linesep \
                       )
            file.close()
             

        
             


# %%
from loaddata import CreateTensorsFrom1Darrays
import os
os.chdir('/home/christian/Documents')

import numpy as np 

dataset='disjunctivevec'


TENSOR_PATH = './datasets/'+dataset+'/tensors/'
  
for  k in range(1,N):
    
    for p in range(1,k+1):
        
         for run in range(NB_samecondi_runs):   
             
                name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)
                
                IMAGEPATH = './datasets/'+dataset+'/images/images' + name + '/'
             
                # tensors names
                sampletensor = 'samples' + name  # choose name of existing or to be created tensor
                labeltensor  = 'labels' + name   # choose name of existing or to be created tensor
                 
                CreateTensorsFrom1Darrays(IMAGEPATH,TENSOR_PATH,sampletensor,labeltensor)            
            
# %%
                
def get_possibilities(N,k,p):



    conjunc=np.math.factorial(k) / ( np.math.factorial(p) * np.math.factorial(k-p))
    p1 = np.math.factorial(k) / ( np.math.factorial(p) * np.math.factorial(k-p))
              
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                