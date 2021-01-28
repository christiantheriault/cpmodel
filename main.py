#%%

import os
from importlib import reload
user = os.environ['HOME']
os.chdir(user+'/Documents')

# CHANGE TO YOUR DEFAULT HOME PATH ACCORDINGLY
#os.chdir('/home/dista/Documents')
# MAKE SURE TO UPDATE YOUR FUNCTIONS PATH
import torch
import numpy
import numpy as np
import sys
sys.path.append(user+'/Documents/cpmodel/')
import torch.nn as nn
import matplotlib.pyplot as plt


import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


# THIS CELL LOADS DATA AND ARCHITECTURE

import loaddata
reload(loaddata)
from loaddata import loadDataTensor
from loaddata import CreateTensorsFrom1Darrays
from loaddata import CreateSoundTensors
from loaddata import CreateImageTensors


import pytz
import copy
from shutil import copyfile

from datetime import datetime

from config import *


#import argparse

# Function to define and load all default parameters
#def parse_args():    
#        
#    # LOAD DEFAULT PARAMETERS
#    N=5
#    netruns=1
#    
#    parser = argparse.ArgumentParser(description='cp model')
#    parser.add_argument('--N', type=int, default=N,help='Number of input dimensions for vector stimuli')
#    parser.add_argument('--netruns', type=int, default=netruns,help='Number of net runs to average')
#    
#    return parser.parse_args()




def main():

    
    
    args = parse_args()
    N=args.N
    NB_ofnetwork_runs=args.netruns    
    
    
    
    codelocation='./cpmodel/'
    #codelocation='/content/drive/My Drive/Colab Notebooks/'
    
    #exclusive
    #dataset='continuous/set2_N1withgeneratecontinuumGOODresultS182139_N6withfixedUjustrials_Nojoker_No_mean_accu'
    dataset='disjunctivevec'
    #dataset='sophie/GOOD_set2_black_insteadof_darkgrey_fixedUfixedposi_CAREFUL_WITHIN_BIAS_DIST_ZERO_PAIRS_no_eval_mode_validation_No_mean_accu'
    
    #dataset2='continuous'
    
    
    codename1 = 'CP_auto_catego.py'
    codename2 = 'loaddata.py'
    codename3 = 'net_classes_and_functions.py'
    codename4 = dataset +'_create_folders.py'  # dataset2
    
    # run info z
    runsname =  'runs_ID_'+str(datetime.now(pytz.timezone('America/Montreal')))[0:19]
    #runsname=runsname.replace(':', '_')
    recordruns=True
    
    reactivaterun=False
    #runsname='runs_ID_2020-03-18 11:34:07'
    #runsname='runs_ID_2020-03-21 18:00:12'
    
    
    
    if recordruns:
        
        CODE_PATH='./datasets/' + dataset + '/' + runsname +'/codes/'
        os.makedirs(CODE_PATH)
        copyfile(codelocation + codename1, CODE_PATH+ codename1 )
        copyfile(codelocation + codename2, CODE_PATH+ codename2 ) 
        copyfile(codelocation + codename3, CODE_PATH+ codename3 ) 
        copyfile(codelocation + codename4, CODE_PATH+ codename4 ) 
    
            
        # Path to where you save your data (outputs, averages, info, etc.)
        
        DATA_PATH = './datasets/' + dataset + '/'+runsname+'/saveddata/'    # change accordingly
        os.makedirs(DATA_PATH)
    
        
        # Path to where nets are saved (for visualization)
        
        NET_PATH = './datasets/' + dataset + '/'+runsname+'/savednet/'
        os.makedirs(NET_PATH)
        
        # Path to where figures are saved
        
        FIGURE_PATH= './datasets/' + dataset + '/'+runsname+'/savedfigures/'
        os.makedirs(FIGURE_PATH)
        
          
        
    else :
      
        if reactivaterun :
          
            CODE_PATH='./datasets/' + dataset + '/'+runsname+'/codes/'
    
            # Path to where you save your data (outputs, averages, info, etc.)
    
            DATA_PATH = './datasets/' + dataset + '/'+runsname+'/saveddata/'    # change accordingly
    
            # Path to where nets are saved (for visualization)
    
            NET_PATH = './datasets/' + dataset + '/'+runsname+'/savednet/'
    
            # Path to where figures are saved
    
            FIGURE_PATH= './datasets/' + dataset + '/'+runsname+'/savedfigures/'
      
        else:  
      
        
            # Path to where you save your tensors
    
            DATA_PATH = './datasets/' + dataset + '/'+ 'testrun/saveddata/'    # change accordingly
    
    
            # Path to where nets are saved (for visualization)
    
            NET_PATH =  './datasets/' + dataset + '/'+ 'testrun/savednet/'
    
            # Path to where figures are saved
    
            FIGURE_PATH=  './datasets/' + dataset + '/'+ 'testrun/savedfigures/'
    
            CODE_PATH='./datasets/' + dataset + '/'  +'/codes/'
    
            copyfile(codelocation + codename1, CODE_PATH+ codename1 )
            copyfile(codelocation + codename2, CODE_PATH+ codename2 ) 
            copyfile(codelocation + codename3, CODE_PATH+ codename3 ) 
            
    #  create tensors
            
    import loaddata
    reload(loaddata)
    import glob
    import gc
    
    percent_use=[1, 1, 1,1,1] 
    nb_of_tensors=[2,2, 1, 1,1]
    
    NB_samecondi_runs=1
    #exclusived
    dataset='disjunctivevec'
    
    r=0
    for run in range(0,NB_samecondi_runs):
    
        for k in range(1,N):
            if k==1:
                pstart=1
            else:
                pstart=1
            
            for p in range(1,k+1):
                
                name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)
                IMAGEPATH = './datasets/' + dataset + '/images/images' + name + '/' # change accordingly
                TENSOR_PATH = './datasets/' + dataset +  '/tensors/'
                
                # tensors names
                sampletensor = 'samples' + name  # choose name of existing or to be created tensor
                labeltensor  = 'labels' + name   # choose name of existing or to be created tensor
    
                print('creating tensor from folder :'  + name    )
    
                CreateTensorsFrom1Darrays(IMAGEPATH,TENSOR_PATH,sampletensor,labeltensor)
    #            CreateImageTensors(IMAGEPATH,TENSOR_PATH,sampletensor,labeltensor)
    #            CreateSoundTensors(IMAGEPATH,TENSOR_PATH,sampletensor,labeltensor,k)
    
    
    
    
    # train
    
    res50_conv=[]            
    
                
    import loaddata
    reload(loaddata)
         
    
    import net_classes_and_functions
    reload(net_classes_and_functions)
    from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp,conv_out_size, squaresize,plotcp
    
    percent_use=[1, 1, 1,1,1] 
    nb_of_tensors=[2,2, 1, 1,1]
    
    
    NB_samecondi_runs=1
#    NB_ofnetwork_runs=1
    
    measurelayer=0
    
    #exclusived
    dataset='disjunctivevec'
    #cptype='_equi'
    cptype='_regular'
    
    
    # To store avg distance for CP evolution on conjunctive/disjunctive case in a loop
    #store_avg_distances=torch.zeros(N-1,N-1,4)
    store_avg_distances_all_runs=torch.zeros(NB_samecondi_runs,NB_ofnetwork_runs,N-1,N-1,4)
    #store_avg_distances_all_runs=torch.load(DATA_PATH+ 'store_avg_distances_all_runs')
    #store_avg_distances_all_runs=torch.tensor([])
    
    
    r=0
    for run in range(0,NB_samecondi_runs):
    
        for k in range(1,N):
            if k==1:
                pstart=1
            else:
                pstart=1
            
            
            for p in range(k,k+1):
                
    
                  
                name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
                IMAGEPATH = './datasets/' + dataset + '/images/images' + name + '/' # change accordingly
                TENSOR_PATH = './datasets/' + dataset +  '/tensors/'
                
                # tensors names
                sampletensor = 'samples' + name  # choose name of existing or to be created tensor
                labeltensor  = 'labels' + name   # choose name of existing or to be created tensor
    
               
    #                name=name+'_?'
                tensorlist=glob.glob(TENSOR_PATH + 'samples' + name )
                labellist=glob.glob(TENSOR_PATH + 'labels' + name )
    
                
                for i in range(NB_ofnetwork_runs):
                           
                      t=0  
                      for tensor in tensorlist :
    
                            sampletensor = 'samples' + tensorlist[t][-len(name):]  
                            labeltensor  = 'labels' +  tensorlist[t][-len(name):] 
                            train_loader ,test_loader, chan, inputsize, usecat = loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)
    
                            outdim=len(usecat)
    
                            # TRAIN AUTENCODERS : MOST PARAMETERS MUST BE SET MANUALLY INSIDE THE CLASS
                            VISUALIZE=True
                            if t==0:
                                load_net=False
                            else :
                                load_net=True
    
                            if load_net :
                                net = STACKEDAUTOENCODER(outdim,chan,inputsize)
                                net.load_state_dict(beforenet.state_dict())
                            else :
                                net = STACKEDAUTOENCODER(outdim,chan,inputsize)
                             
    
    
    #                        net.layers[0].encode[3]=nn.Dropout2d(p=0.0)    
    #                        net.layers[1].encode[3]=nn.Dropout2d(p=0.0)
                            
                                
    #                        net.layers[0].encode[3]=nn.Dropout(p=0.0)    
    #                        net.layers[1].encode[3]=nn.Dropout(p=0.9)
    #                        net.layers[2].encode[3]=nn.Dropout(p=0.0)
    #                        net.layers[3].encode[3]=nn.Dropout(p=0.0)
                            
    #                        net.layers[0].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[1].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[2].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[3].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[2].encode[2]=nn.LeakyReLU(1)
    
    
    
                            print('TRAINING ON ' + 'tensor_' + sampletensor + '   netrun='   + str(i))
                            fullname=name + '_netrun_'   + str(i)
                            beforenet=trainauto(fullname,VISUALIZE,NET_PATH,net,res50_conv,train_loader,test_loader,global_train=False)
          
                            
                            
    ##                        FOR PCA
    #                        cat=train_loader.dataset.tensors[0]
    #                        cat=np.squeeze(cat.data.numpy())
    #                        mean=np.mean(cat,axis=0)
    #                        cat=cat-np.matlib.repmat(mean,np.shape(cat)[0],1)
    #                        cov=np.matmul(np.transpose(cat,[1,0]),cat)
    #                        lambd, v =np.linalg.eig(cov)
    #                        lambd[lambd.real<10**-3]=0
    ##                        lambd=lambd.real/max(lambd.real)*4
    #                        lambd[lambd>10**-3]=1
    ##
    #                        v=np.dot(v,np.diag(lambd))
    #            #            w, v =scipy.linalg.eigh(cov)
    ##                        v=torch.tensor(v.real)
    #                        v=torch.tensor(np.transpose(v.real,[1,0]))
    ##                        v=torch.eye(N)
    #
    #                        beforenet = STACKEDAUTOENCODER(outdim,chan,inputsize)
    #                        beforenet.layers[0].encode[0].weight=torch.nn.Parameter(v)
                            
                            
                       
                            
                            t=t+1
                            del train_loader ,test_loader
    
    
    
                      t=0
                      for tensor in tensorlist :
    
                            sampletensor = 'samples' + tensorlist[t][-len(name):]  
                            labeltensor  = 'labels' +  tensorlist[t][-len(name):] 
                            train_loader ,test_loader, chan, inputsize, usecat = loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)
    
                            outdim=len(usecat)
    
    
                            # TRAIN CLASSIFIER :  : MOST PARAMETERS MUST BE SET MANUALLY INSIDE THE CLASS
    
                            if t==0:
                                net = type(beforenet)(outdim,chan,inputsize) # get a new instance
                                net.load_state_dict(beforenet.state_dict())
                            else:
                                net = type(classifier)(outdim,chan,inputsize) # get a new instance
                                net.load_state_dict(classifier.state_dict())
                                
    #                        net.layers[0].encode[3]=nn.Dropout(p=0.0)    
    #                        net.layers[1].encode[3]=nn.Dropout(p=0.0)
    #                        net.layers[2].encode[3]=nn.Dropout(p=0.0)
    #                        net.layers[3].encode[3]=nn.Dropout(p=0.0)
     
        
    #                        net.layers[0].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[1].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[2].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[3].encode[2]=nn.LeakyReLU(1)
    #                        net.layers[4].encode[2]=nn.LeakyReLU(1)
    
    #                        net.layers[4].encode[2]=nn.Tanh()
    #                        net.layers[4].encode[2]=nn.Tanh()
    #                        net.layers[2].encode[2]=nn.Tanh()
                            
        
    
                            print('TRAINING ON ' + 'tensor  ' + sampletensor + '   netrun='   + str(i))
                            fullname=name + '_netrun_'   + str(i)
                            classifier=trainclassifier(fullname,VISUALIZE,NET_PATH,net,res50_conv,inputsize,train_loader,test_loader)
                            t=t+1
                            del train_loader ,test_loader
      
                    
    ##                  beforenet.layers[0].encode[0].weight=torch.nn.Parameter(torch.eye(3))
    #                  dists=torch.zeros(len(tensorlist),4)
    #                  t=0
    #                  for tensor in tensorlist :                          
    #                        sampletensor = 'samples' + tensorlist[t][-len(name):]  
    #                        labeltensor  = 'labels' +  tensorlist[t][-len(name):] 
    #                        train_loader ,test_loader, chan, inputsize, usecat = loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)
    #                        
    #                        loader=test_loader
    #                        autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin=\
    #                        computecp(usecat,beforenet,classifier,loader,measurelayer=measurelayer)
    #                        dists[t,:]=torch.tensor([autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])     
    #                        t=t+1
    #                  autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin= \
    #                  torch.mean(dists,dim=0)
    #                  
    #                  
    #                  store_avg_distances_all_runs[r,i,k-1,p-1,:]=\
    #                  torch.tensor([autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])
    #                  torch.save(store_avg_distances_all_runs, DATA_PATH+ 'store_avg_distances_all_runs'+ cptype)
    
                      
                      
                        
                      torch.save(beforenet.state_dict(), NET_PATH+'beforenet' +name +'_netrun'  +str(i) )
                      torch.save(classifier.state_dict(), NET_PATH+'classifier' +name+'_netrun'  +str(i))
                
                gc.collect()   
        r=r+1
                
    
    
    # Just CP : by reloading saved nets
            
    import glob
    import gc
    
    import net_classes_and_functions
    reload(net_classes_and_functions)
    from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp,conv_out_size, squaresize,plotcp
    
    percent_use=[1, 1, 1,1,1] 
    nb_of_tensors=[2,2, 1, 1,1]
    
    
    NB_samecondi_runs=1
#    NB_ofnetwork_runs=1
    measurelayer=0
    
    #datasetorigin='continuous'
    #datasetorigin='disjunctivevec_nonlinsep'
    dataset='disjunctivevec'
    
    
    
    #cptype='_equi'
    cptype='_regular'
    #tensors= '/tensors'+cptype+'/'
    tensors= '/tensors/'
    avgname='store_avg_distances_all_runs'+cptype
    
    
    # To store avg distance for CP evolution on conjunctive/disjunctive case in a loop
    #store_avg_distances=torch.zeros(N-1,N-1,4)
    store_avg_distances_all_runs=torch.zeros(NB_samecondi_runs,NB_ofnetwork_runs,N-1,N-1,4)
    #store_avg_distances_all_runs=torch.load(DATA_PATH+ 'store_avg_distances_all_runs')
    
    #store_avg_distances_all_runs=torch.tensor([])
    
    r=0
    for run in range(0,NB_samecondi_runs):
    
        for k in range(1,N):
            
            if k==1:
                pstart=1
            else:
                pstart=1        
            
            for p in range(k,k+1):
                
                  
                name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
                TENSOR_PATH = './datasets/' + dataset +  tensors
                
                # tensors names
                sampletensor = 'samples' + name  # choose name of existing or to be created tensor
                labeltensor  = 'labels' + name   # choose name of existing or to be created tensor
    
               
    #            Name=name+'_*'
                Name=name
                tensorlist=sorted(glob.glob(TENSOR_PATH + 'samples' + Name ))
                labellist=sorted(glob.glob(TENSOR_PATH + 'labels' + Name ))
                
    
                
                for i in range(NB_ofnetwork_runs):
                    
                    
                      
    
                      
                      dists=torch.zeros(len(tensorlist),4)
                      t=0
                      for tensor in tensorlist :                          
                            sampletensor = tensorlist[t][len(TENSOR_PATH):]  
                            labeltensor  = labellist[t][len(TENSOR_PATH):] 
                            print('COMPUTING CP ON ' + 'tensor  ' + sampletensor + '   netrun='   + str(i))
                            train_loader ,test_loader, chan, inputsize, usecat = loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)
                            
                            
                            outdim=len(usecat)
                            
                            beforenet = STACKEDAUTOENCODER(outdim,chan,inputsize)
                            beforenet.load_state_dict(torch.load(NET_PATH+'beforenet'+ name+'_netrun'  +str(i)))
                            
    #                        beforenet.layers[0].encode[3]=nn.Dropout(p=0.0)    
    #                        beforenet.layers[1].encode[3]=nn.Dropout(p=0.0)
    #                        beforenet.layers[2].encode[3]=nn.Dropout(p=0.0)
    #                        beforenet.layers[3].encode[3]=nn.Dropout(p=0.0)
    
    #                        beforenet.layers[0].encode[2]=nn.LeakyReLU(1)
    #                        beforenet.layers[1].encode[2]=nn.LeakyReLU(1)
    #                        beforenet.layers[2].encode[2]=nn.LeakyReLU(1)
    #                        beforenet.layers[3].encode[2]=nn.LeakyReLU(1)
                            
                            
                            classifier = type(beforenet)(outdim,chan,inputsize)
                            classifier.load_state_dict(torch.load(NET_PATH+'classifier'+ name+'_netrun'  +str(i)))
    
    #                        classifier.layers[0].encode[3]=nn.Dropout(p=0.0)    
    #                        classifier.layers[1].encode[3]=nn.Dropout(p=0.0)
    #                        classifier.layers[2].encode[3]=nn.Dropout(p=0.0)
    #                        classifier.layers[3].encode[3]=nn.Dropout(p=0.0)
    
    
    #                        classifier.layers[0].encode[2]=nn.LeakyReLU(1)
    #                        classifier.layers[1].encode[2]=nn.LeakyReLU(1)
    #                        classifier.layers[1].encode[2]=nn.Tanh()
    #                        classifier.layers[3].encode[2]=nn.LeakyReLU(1)                     
                            
    
                            loader=test_loader
                            autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin=\
                            computecp(usecat,beforenet,classifier,res50_conv,loader,measurelayer=measurelayer)
                            dists[t,:]=torch.tensor([autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])     
                            t=t+1
                      autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin= \
                      torch.mean(dists,dim=0)
    #                  print(dists)
                      
                  
                      store_avg_distances_all_runs[r,i,k-1,p-1,:]=\
                      torch.tensor([autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])
                      torch.save(store_avg_distances_all_runs, DATA_PATH+ avgname)
    
                    
                gc.collect()   
        r=r+1
    
    
    
    
    
    
    
                           
    
    #
    # plot CP with k and p
    
    import numpy as np
    import matplotlib.pyplot as plt
    import net_classes_and_functions
    reload(net_classes_and_functions)
    from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp,conv_out_size, squaresize,plotcp
    
    #cptype='_equi'
    cptype='_regular'
    
    avgname='store_avg_distances_all_runs'+cptype #
    
    store_avg_distances_all_runs=torch.load(DATA_PATH+avgname)
    
    # reshape into (netruns * condiruns , k, p, 4)
    store_avg_distances_all_runs=store_avg_distances_all_runs.view(-1, *(store_avg_distances_all_runs.size()[2:]))  
    
    
    
    GLOBALCP=torch.zeros(N-1,N-1)
    COMPRESSION=torch.zeros(N-1,N-1)
    SEPARATION=torch.zeros(N-1,N-1)
    INITIAL_SEPARATION=torch.zeros(N-1,N-1)
    BEFORE_BETWEEN=torch.zeros(N-1,N-1)
    AFTER_BETWEEN=torch.zeros(N-1,N-1)
    BEFORE_WITHIN=torch.zeros(N-1,N-1)
    AFTER_WITHIN=torch.zeros(N-1,N-1)
    
    for k in range(0,N-1):
        
        for p in range(0,k+1):
          
            meanonruns=torch.mean(store_avg_distances_all_runs[:,k,p,:],dim=0)
    
            #meanonruns=torch.mean(store_avg_distances_all_runs[:,:,k,p,:],dim=0)
            #mean=torch.mean(meanonruns[:,k,p,:],dim=0)
            autobetween,autowithin,classibetween,classiwithin = [*meanonruns]
            sepa=classibetween-autobetween
            compre=classiwithin-autowithin
            
            GLOBALCP[k,p] = sepa-compre
            COMPRESSION[k,p]=compre
            SEPARATION[k,p]=sepa
            BEFORE_BETWEEN[k,p]=autobetween
            AFTER_BETWEEN[k,p]=classibetween
            BEFORE_WITHIN[k,p]=autowithin
            AFTER_WITHIN[k,p]=classiwithin
            
            INITIAL_SEPARATION[k,p] = autobetween-autowithin
            
            
            plt.close()
            plotcp(autobetween,autowithin,classibetween,classiwithin)
            
    
            plt.savefig(FIGURE_PATH +'cp'  +'_N=' + str(N) + '_K=' + str(k+1) + '_P=' + str(p+1) +cptype+ '.png')
    
    
            
            
    torch.save(GLOBALCP, DATA_PATH + 'GLOBALCP'+cptype)
    torch.save(COMPRESSION, DATA_PATH + 'COMPRESSION'+cptype)
    torch.save(SEPARATION,DATA_PATH + 'SEPARATION'+cptype)
    torch.save(INITIAL_SEPARATION,DATA_PATH + 'INITIAL_SEPARATION'+cptype)
    
    torch.save(BEFORE_BETWEEN, DATA_PATH + 'BEFORE_BETWEEN'+cptype)
    torch.save(AFTER_BETWEEN, DATA_PATH + 'AFTER_BETWEEN'+cptype)
    torch.save(BEFORE_WITHIN,DATA_PATH + 'BEFORE_WITHIN'+cptype)
    torch.save(AFTER_WITHIN,DATA_PATH + 'AFTER_WITHIN'+cptype)
    
    
    
    # CP curves
    
    import matplotlib.pyplot as plt
    import numpy as np    
    
    
    #cptype='_equi'
    cptype='_regular'
    
    diag=0
            
    GLOBALCP=torch.load(DATA_PATH + 'GLOBALCP'+cptype)
    COMPRESSION=torch.load(DATA_PATH + 'COMPRESSION'+cptype)
    SEPARATION=torch.load(DATA_PATH + 'SEPARATION'+cptype)
    INITIAL_SEPARATION=torch.load(DATA_PATH + 'INITIAL_SEPARATION'+cptype)
    
    BEFORE_BETWEEN=torch.load(DATA_PATH + 'BEFORE_BETWEEN'+cptype)
    AFTER_BETWEEN=torch.load(DATA_PATH + 'AFTER_BETWEEN'+cptype)
    BEFORE_WITHIN=torch.load(DATA_PATH + 'BEFORE_WITHIN'+cptype)
    AFTER_WITHIN=torch.load(DATA_PATH + 'AFTER_WITHIN'+cptype)
    
    
    #GLOBALCP=torch.load(DATA_PATH + 'GLOBALCP')
    #COMPRESSION=torch.load(DATA_PATH + 'COMPRESSION')
    #SEPARATION=torch.load(DATA_PATH + 'SEPARATION')
    #INITIAL_SEPARATION=torch.load(DATA_PATH + 'INITIAL_SEPARATION')
    
    #BEFORE_BETWEEN=torch.load(DATA_PATH + 'BEFORE_BETWEEN')
    #AFTER_BETWEEN=torch.load(DATA_PATH + 'AFTER_BETWEEN')
    #BEFORE_WITHIN=torch.load(DATA_PATH + 'BEFORE_WITHIN')
    #AFTER_WITHIN=torch.load(DATA_PATH + 'AFTER_WITHIN')
    
    
    
    
    ## for equi distance : can only use up to N/2
    #use=int(N/2)
    #GLOBALCP=GLOBALCP[0:use,0:use]
    #COMPRESSION=COMPRESSION[0:use,0:use]
    #SEPARATION=SEPARATION[0:use,0:use]
    #INITIAL_SEPARATION=INITIAL_SEPARATION[0:use,0:use]
    #numberval=GLOBALCP.size(0)+1
    #valnames= np.arange(1,GLOBALCP.size(0)+1)
    
    
    numberval=N-diag
    valnames= np.arange(1,N-diag)
    
    
    
    
    plt.figure()
    fig, ax = plt.subplots()
    
    #plt.plot(numpy.flip(torch.diag(SEPARATION-COMPRESSION).data.numpy()),linewidth=4,marker='.',markersize=20)
    plt.plot(torch.diag(SEPARATION-COMPRESSION,diagonal=diag).data.numpy(),linewidth=4,marker='.',markersize=20)
    #plt.plot(SEPARATION[:,0].data.numpy()-COMPRESSION[:,0].data.numpy(),linewidth=4,marker='.',markersize=20)
    
    
    plt.ylabel("Global CP  ",fontsize=15)
    plt.xlabel("K",fontsize=15)
    plt.title('Conjunctive',fontsize=15)
    ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
    #ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval)))) 
    ax.xaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval))) 
    
    
    #plt.show()
    
    plt.savefig(FIGURE_PATH +'globalcp_K=P'+cptype +'.png')
    
    plt.figure()
    fig, ax = plt.subplots()
    
    #plt.plot(numpy.flip(torch.diag(COMPRESSION).data.numpy()),linewidth=4,marker='.',markersize=20, color='green')
    plt.plot(torch.diag(COMPRESSION,diagonal=diag).data.numpy(),linewidth=4,marker='.',markersize=20, color='green')
    #plt.plot(COMPRESSION[:,0].data.numpy(),linewidth=4,marker='.',markersize=20)
    
    
    
    plt.ylabel("Compression  ",fontsize=15)
    plt.xlabel("K",fontsize=15)
    plt.title('Conjunctive',fontsize=15) 
    ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
    #ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
    
    
    #plt.show()
    
    plt.savefig(FIGURE_PATH +'Compression_K=P' +cptype+ '.png')
    
    
    plt.figure()
    fig, ax = plt.subplots()
    
    #plt.plot(numpy.flip(torch.diag(SEPARATION).data.numpy()),linewidth=4,marker='.',markersize=20, color='blue')
    plt.plot(torch.diag(SEPARATION,diagonal=diag).data.numpy(),linewidth=4,marker='.',markersize=20, color='blue')
    #plt.plot(SEPARATION[:,0].data.numpy(),linewidth=4,marker='.',markersize=20)
    
    
    
    plt.ylabel("Separation ",fontsize=15)
    plt.xlabel("K",fontsize=15)
    plt.title('Conjunctive',fontsize=15) 
    ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
    #ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
    
    
    #plt.show()
    
    plt.savefig(FIGURE_PATH +'Separation_K=P'+cptype +'.png')
    
    
    
    plt.figure()
    fig, ax = plt.subplots()
    
    #plt.plot(numpy.flip(torch.diag(INITIAL_SEPARATION).data.numpy()),linewidth=4,marker='.',markersize=20, color='blue')
    plt.plot(torch.diag(INITIAL_SEPARATION,diagonal=diag).data.numpy(),linewidth=4,marker='.',markersize=20, color='blue')
    #plt.plot(INITIAL_SEPARATION[:,0].data.numpy(),linewidth=4,marker='.',markersize=20)
    
    
    
    
    plt.ylabel("Initial separation ",fontsize=15)
    plt.xlabel("K",fontsize=15)
    plt.title('Conjunctive',fontsize=15) 
    ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
    #ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
    
    
    #plt.show()
    
    plt.savefig(FIGURE_PATH +'Initial_Separation_K=P'+cptype +'.png')
    
    
    
    plt.figure()
    fig, ax = plt.subplots()
    
    #im=plt.plot(numpy.flip(torch.diag(AFTER_BETWEEN).data.numpy()),'blue',\
    #            numpy.flip(torch.diag(BEFORE_BETWEEN).data.numpy()),'skyblu1e',linewidth=4,marker='.',markersize=20)
    
    im=plt.plot(torch.diag(AFTER_BETWEEN,diagonal=diag).data.numpy(),'blue',\
                torch.diag(BEFORE_BETWEEN,diagonal=diag).data.numpy(),'skyblue',linewidth=4,marker='.',markersize=20)
    
    
    #im=plt.plot(AFTER_BETWEEN[:,0].data.numpy(),'blue',\
    #            BEFORE_BETWEEN[:,0].data.numpy(),'skyblue',linewidth=4,marker='.',markersize=20)
    
    
    
    
    plt.legend([im[0],im[1]],['After','Before'],bbox_to_anchor=(0.6,0.6),fontsize=15)
    plt.ylabel("Between ",fontsize=15)
    plt.xlabel("K",fontsize=15)
    plt.title('Conjunctive',fontsize=15) 
    ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
    #ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
    
    
    #plt.show()
    
    plt.savefig(FIGURE_PATH +'between_K=P'+cptype +'.png')
    
    
    
    
    plt.figure()
    fig, ax = plt.subplots()
    
    #im=plt.plot(numpy.flip(torch.diag(AFTER_WITHIN).data.numpy()),'green',\
    #            numpy.flip(torch.diag(BEFORE_WITHIN).data.numpy()),'lightgreen',linewidth=4,marker='.',markersize=20)
    
    im=plt.plot(torch.diag(AFTER_WITHIN,diagonal=diag).data.numpy(),'green',\
                torch.diag(BEFORE_WITHIN,diagonal=diag).data.numpy(),'lightgreen',linewidth=4,marker='.',markersize=20)
    
    #im=plt.plot(AFTER_WITHIN[:,0].data.numpy(),'blue',\
    #            BEFORE_WITHIN[:,0].data.numpy(),'skyblue',linewidth=4,marker='.',markersize=20)
    
    
    
    plt.legend([im[0],im[1]],['After','Before'],bbox_to_anchor=(0.6,0.6),fontsize=15)
    
    plt.ylabel("Within ",fontsize=15)
    plt.xlabel("K",fontsize=15)
    plt.title('Conjunctive',fontsize=15) 
    ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
    #ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
    ax.xaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
    
    
    #plt.show()
    
    plt.savefig(FIGURE_PATH +'After_K=P'+cptype +'.png')
    
    
if __name__ == "__main__":
    main()    
    
