#%%

import os
from importlib import reload

# CHANGE TO YOUR DEFAULT HOME PATH ACCORDINGLY
os.chdir('/home/dista/Documents')
# MAKE SURE TO UPDATE YOUR FUNCTIONS PATH
import torch
import numpy
import numpy as np
import sys
sys.path.append('/home/dista/Documents/pytorch_codes/')
import torch.nn as nn
import matplotlib.pyplot as plt


import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim




#%% 
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


codelocation='./pytorch_codes/'
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
# %% create tensors
        
import loaddata
reload(loaddata)
from loaddata import loadDataTensor
from loaddata import CreateTensorsFrom1Darrays
from loaddata import CreateSoundTensors
from loaddata import CreateImageTensors
import glob
import gc

percent_use=[1, 1, 1,1,1] 
nb_of_tensors=[2,2, 1, 1,1]

N=5
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

#nameslist=np.load('./datasets/'+'textures_michel'+'/info/' + 'nameslist.npy')
#nameslist[3:9]   
# %% import resnet

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model = models.resnet50(pretrained=True).to(device)
# %% import resnet get only conv layers

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#res50_model = models.resnet34(pretrained=True).to(device)
#res50_conv = nn.Sequential(*list(res50_model.children())[:-1])
#for param in res50_conv.parameters():
#    param.requires_grad = False
#res50_model.fc
            
res50_conv=[]            
## %%
#import glob
#
#N=8
#k=1
#p=1
#r=0 
#name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
#TENSOR_PATH = './datasets/' + dataset +  '/tensors/'
#            
## tensors names
#sampletensor = 'samples' + name  # choose name of existing or to be created tensor
#abeltensor  = 'labels' + name   # choose name of existing or to be created tensor
#
#           
##                name=name+'_?'
#tensorlist=glob.glob(TENSOR_PATH + 'samples' + name )
#labellist=glob.glob(TENSOR_PATH + 'labels' + name )
#t=0  
#for tensor in tensorlist :
#
#    sampletensor = 'samples' + tensorlist[t][-len(name):]  
#    labeltensor  = 'labels' +  tensorlist[t][-len(name):] 
#    train_loader ,test_loader, chan, inputsize, usecat = loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)
# %%

#inputs, labels = next(iter(train_loader))
##inputs, labels = Variable(inputs), Variable(labels)
#labels=labels.to(device)
#inputs=inputs.to(device)
#outputs = res50_conv(inputs)
#outputs.size()    
    

#%% train
            
import loaddata
reload(loaddata)
from loaddata import loadDataTensor       
     
import glob
import gc


import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp,conv_out_size, squaresize,plotcp

percent_use=[1, 1, 1,1,1] 
nb_of_tensors=[2,2, 1, 1,1]


N=5
NB_samecondi_runs=1
NB_ofnetwork_runs=1

measurelayer=1

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
            


#%% Just CP : by reloading saved nets
        
import glob
import gc

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp,conv_out_size, squaresize,plotcp

percent_use=[1, 1, 1,1,1] 
nb_of_tensors=[2,2, 1, 1,1]


N=5
NB_samecondi_runs=1
NB_ofnetwork_runs=1
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
for run in range(0,NB_ofnetwork_runs):

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







                       

#%% 
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


N=5

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



#%% CP curves

N=5
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





#%% CP density plot
N=8
import matplotlib
import matplotlib.pyplot as plt
import copy

#cptype='_equi'
cptype='_regular'


import numpy as np
GLOBALCP=torch.load(DATA_PATH + 'GLOBALCP'+cptype)
COMPRESSION=torch.load(DATA_PATH + 'COMPRESSION'+cptype)

SEPARATION=torch.load(DATA_PATH + 'SEPARATION'+cptype)
INITIAL_SEPARATION=torch.load(DATA_PATH + 'INITIAL_SEPARATION'+cptype)

BEFORE_BETWEEN=torch.load(DATA_PATH + 'BEFORE_BETWEEN'+cptype)
AFTER_BETWEEN=torch.load(DATA_PATH + 'AFTER_BETWEEN'+cptype)
BEFORE_WITHIN=torch.load(DATA_PATH + 'BEFORE_WITHIN'+cptype)
AFTER_WITHIN=torch.load(DATA_PATH + 'AFTER_WITHIN'+cptype)


## for equi distance : can only use up to N/2
#use=int(N/2)
#GLOBALCP=GLOBALCP[0:use,0:use]
#COMPRESSION=COMPRESSION[0:use,0:use]
#SEPARATION=SEPARATION[0:use,0:use]
#INITIAL_SEPARATION=INITIAL_SEPARATION[0:use,0:use]
#numberval=GLOBALCP.size(0)+1
#valnames= np.arange(1,GLOBALCP.size(0)+1)

#
numberval=N
valnames= np.arange(1,N)


plt.figure()
fig, ax = plt.subplots()

GLOBALCP = np.ma.array(GLOBALCP, mask=(GLOBALCP==0))
cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
cmap1.set_bad('white',1.)
cmap2.set_bad('white',1.)

im2 = plt.imshow(np.flip(GLOBALCP,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-GLOBALCP,axis=1), cmap=cmap1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.title('GLOBALCP',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'globalcp'+cptype +'.png')

plt.show()


plt.figure()
fig, ax = plt.subplots()

COMPRESSION = np.ma.array(COMPRESSION, mask=(COMPRESSION==0))
#cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
#cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
#cmap1.set_under('white',1.)
#cmap2.set_under('white',1.)

im2 = plt.imshow(np.flip(COMPRESSION,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-COMPRESSION,axis=1), cmap=cmap1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

plt.title('COMPRESSION',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'compression'+cptype +'.png')

plt.show()




plt.figure()
fig, ax = plt.subplots()

SEPARATION = np.ma.array(SEPARATION, mask=(SEPARATION==0))
#cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
#cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
#cmap1.set_under('white',1.)
#cmap2.set_under('white',1.)

im2 = plt.imshow(np.flip(SEPARATION,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-SEPARATION,axis=1), cmap=cmap1)

ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)
plt.title('SEPARATION',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'separation'+cptype +'.png')

plt.show()




plt.figure()
fig, ax = plt.subplots()


INITIAL_SEPARATION = np.ma.array(INITIAL_SEPARATION, mask=(INITIAL_SEPARATION==0))
#cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
#cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
#cmap1.set_under('white',1.)
#cmap2.set_under('white',1.)


im2 = plt.imshow(np.flip(INITIAL_SEPARATION,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-INITIAL_SEPARATION,axis=1), cmap=cmap1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

plt.title('INITIAL_SEPARATION',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'initial_separation' +cptype+'.png')

plt.show()


plt.figure()
fig, ax = plt.subplots()


BEFORE_BETWEEN = np.ma.array(BEFORE_BETWEEN, mask=(BEFORE_BETWEEN==0))
#cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
#cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
#cmap1.set_under('white',1.)
#cmap2.set_under('white',1.)


im2 = plt.imshow(np.flip(BEFORE_BETWEEN,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-BEFORE_BETWEEN,axis=1), cmap=cmap1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

plt.title('BEFORE_BETWEEN',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'BEFORE_BETWEEN' +cptype+'.png')

plt.show()


AFTER_BETWEEN = np.ma.array(AFTER_BETWEEN, mask=(AFTER_BETWEEN==0))
#cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
#cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
#cmap1.set_under('white',1.)
#cmap2.set_under('white',1.)


im2 = plt.imshow(np.flip(AFTER_BETWEEN,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-AFTER_BETWEEN,axis=1), cmap=cmap1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

plt.title('AFTER_BETWEEN',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'AFTER_BETWEEN' +cptype+'.png')

plt.show()



AFTER_WITHIN = np.ma.array(AFTER_WITHIN, mask=(AFTER_WITHIN==0))
#cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
#cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
#cmap1.set_under('white',1.)
#cmap2.set_under('white',1.)


im2 = plt.imshow(np.flip(AFTER_WITHIN,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-AFTER_WITHIN,axis=1), cmap=cmap1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

plt.title('AFTER_WITHIN',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'AFTER_WITHIN' +cptype+'.png')

plt.show()




BEFORE_WITHIN = np.ma.array(BEFORE_WITHIN, mask=(BEFORE_WITHIN==0))
#cmap1= copy.copy(matplotlib.cm.get_cmap('viridis'))
#cmap2= copy.copy(matplotlib.cm.get_cmap('viridis_r'))
#cmap1.set_under('white',1.)
#cmap2.set_under('white',1.)


im2 = plt.imshow(np.flip(BEFORE_WITHIN,axis=1), cmap=cmap2)
im = plt.imshow(np.flip(-BEFORE_WITHIN,axis=1), cmap=cmap1)
ax.xaxis.set_tick_params(labelsize=18)
ax.yaxis.set_tick_params(labelsize=18)

plt.title('BEFORE_WITHIN',fontsize=15)
ax.xaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.flip(range(1,numberval))))
ax.yaxis.set_major_locator(plt.FixedLocator(range(0,numberval-1)))
ax.yaxis.set_major_formatter(plt.FixedFormatter(range(1,numberval)))
plt.ylabel("K ",fontsize=18)
plt.xlabel("P",fontsize=18)
plt.colorbar(im2, orientation='vertical')
plt.gcf().subplots_adjust(bottom=0.15)
plt.savefig(FIGURE_PATH +'BEFORE_WITHIN' +cptype+'.png')

plt.show()

#%% 
# VISUALIZATION :RECONSTRUCTION

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import visualize_reconstruction

# RECONSTRUCTION

network=beforenet 
visualize_reconstruction(network,fulltrain_loader) 



#%% 
#VISUALIZATION : FIRST LAYER FILTERS


import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import visualize_firstlayer_filters

# VISUALIZE FILTERS ON FIRST LAYER : FOR CONVOLUTIONAL NET ONLY

network=beforenet 
visualize_firstlayer_filters(network) 


#%% 
#VISUALIZE TOP LEVELS FEATURES
# BACKPROPAGATION ON PIXEL VALUE
# CHOOSE LAYER AND FILTER (UNIT)

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import visualize_top_learned_features

network=classifier
#network=beforenet    

encodelayer=3
filterid=0       
visualize_top_learned_features(network,fulltrain_loader,encodelayer=encodelayer,filterid=filterid)

#%% 
#VISUALIZE LAYER EVOLUTION
# DEFAULT/LOGICAL IS LAST LAYER BEFORE CATEGORIZATION

import loaddata
reload(loaddata)
from loaddata import loadDataTensor       
     
import glob
import gc


import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import visualize_layer








r=0
NB_samecondi_runs=1
NB_ofnetwork_runs=1
N=7
layerid=1


for run in range(0,NB_samecondi_runs):

    for k in range(5,6):
        if k==1:
            pstart=1
        else:
            pstart=1
        
        
        for p in range(5,6):
            
              
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




                        fullname=name + '_netrun_'   + str(i)
                        beforenetlist=torch.load(NET_PATH + 'beforenetlayerlist' + fullname+ '.pt' )
                        classifierlist=torch.load(NET_PATH + 'classifierlist' + fullname+ '.pt' )
                        
#                        beforenetlist=[]
#                        classifierlist=[]
                        loader=test_loader
                        ani=visualize_layer(loader,res50_conv, beforenetlist, classifierlist, layerid=layerid)
                        
                        
                        ani.save(FIGURE_PATH + 'layeranimation' + fullname + '.mp4')
                        
                        del beforenetlist, classifierlist



#%% 
#VISUALIZE CP EVOLUTION
# DEFAULT/LOGICAL IS LAST LAYER BEFORE CATEGORIZATION

dataset='textures_lucas'

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import visualize_cp


r=0
NB_samecondi_runs=1
NB_ofnetwork_runs=1
N=8
layerid=1


for run in range(0,NB_samecondi_runs):

    for k in range(1,2):
        if k==1:
            pstart=1
        else:
            pstart=1
        
        
        for p in range(k,k+1):
            
            name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)


            for i in range(NB_ofnetwork_runs):
                

                        fullname=name + '_netrun_'   + str(i)
                        beforenetlist=torch.load(NET_PATH + 'beforenetlayerlist' + fullname+ '.pt' )
                        classifierlist=torch.load(NET_PATH + 'classifierlist' + fullname+ '.pt' )

#                        beforenetlist=torch.load(NET_PATH + 'beforenetlayerlist.pt' )
#                        classifierlist=torch.load(NET_PATH + 'classifierlist.pt' )
                        
                        
                        ani=visualize_cp(usecat,test_loader, beforenetlist, classifierlist, layerid=layerid)

FIGURE_PATH = './datasets/'+dataset+'/savedfigures/'

ani.save(FIGURE_PATH + 'CPanimation' + fullname+ '.mp4')

del beforenetlist, classifierlist



#%%
#  CHECK OUTPUT
# SIMPLE DISPLAY OF TARGETS AND PREDICTUED VALUES

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import checkclassoutput


checkclassoutput(classifier,test_loader,numberoftestsamples=20)    



#%% 
#ONLY FOR SMALL FULLLY CONNECTED 

# VISUALIZE WEIGHT ASSIGNED TO UNITTS AT A CHOSEN LAYER :

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import STACKEDAUTOENCODER
from net_classes_and_functions import visualize_input_dimensions

N=8
layer=0

NB_samecondi_runs=1
NB_network_runs=1

W=torch.zeros(NB_samecondi_runs*NB_network_runs,N,N)
#W=torch.zeros(NB_samecondi_runs*NB_network_runs,100,100)
#W=torch.zeros(NB_samecondi_runs*NB_network_runs,100,3)



j=0

for k in range(6,7):
    p=k
    for r in range(NB_samecondi_runs):
        for i in range(NB_network_runs):

        
        
                name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
                outdim=2
                chan=1
                inputsize=[1,1]
                beforenet = STACKEDAUTOENCODER(outdim,chan,inputsize)
                beforenet.load_state_dict(torch.load(NET_PATH+'beforenet'+ name+'_netrun'  +str(i)))
        #        beforenet.layers[0].encode[3]=nn.Dropout(p=0.9)
                classifier = type(beforenet)(outdim,chan,inputsize)
                classifier.load_state_dict(torch.load(NET_PATH+'classifier'+ name+'_netrun'  +str(i)))
        #        classifier.layers[0].encode[3]=nn.Dropout(p=0.6) 
                beforenet.eval()
                classifier.eval()
                
                
                
                # choose which net : autoencoder or categorizer
#                network=classifier
                network=beforenet
                W[i,:,:]=network.layers[layer].encode[0].weight
#                visualize_input_dimensions(network,layer=0)
                
                j=j+1



    bar=visualize_input_dimensions(torch.mean(W,dim=0))
#    covposi=[13, 15,  6,  9, 11,  7, 26, 16,  0, 12]
#    covposi=[ 5, 18, 19,  3,  2,  6]
    covposi=[0, 5, 1, 2, 6, 7]
    for i in range(len(covposi)):
        bar[covposi[i]].set_color('red')
    plt.savefig(FIGURE_PATH +'auto_drop0'+'.png')


    W=torch.mean(W,dim=0).data.numpy()
    w=np.sqrt(np.sum(W**2,0))
    print(w)
    print(np.sqrt(np.sum(w[0:7]**2,0)))


#%% 
#compare disjunctive with sophie's
    

import numpy as np

dataset='disjunctivevec'

k=6
p=6
run=0

name= '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)
IMAGEPATH = './datasets/' + dataset + '/images/images' + name + '/' # change accordingly
TENSOR_PATH = './datasets/' + dataset +  '/tensors/'

#A=np.load(IMAGEPATH + 'cat0/' + 'A.npy')
# tensors names
sampletensor = 'samples' + name  # choose name of existing or to be created tensor
labeltensor  = 'labels' + name   # choose name of existing or to be created tensor

train_loader ,test_loader, chan, inputsize, usecat = \
loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)



#train_loader, fulltrain_loader2 ,test_loader, chan, inputsize, usecat=\
#loadSophie_vec(TENSOR_PATH,sampletensor,labeltensor,N,k=k,p=p,nb=nb)    


A=np.load(IMAGEPATH + 'cat0/' + 'A.npy')
B=np.load(IMAGEPATH + 'cat1/' + 'B.npy')


print(B[0:30])
train_loader.dataset.tensors[0][-30:-1]


#%% 
# test equi code :  SEE CORRESPONDING PY FILE FOR MORE RECENT

# K and d must be equal of less than N/2
# d must be more or equal to K   

import loaddata
reload(loaddata)
from loaddata import CreateEquidistantTensors, loadDataTensor


dataset='equidistant'

TENSOR_PATH = './datasets/' + dataset +  '/tensors/'


N=8
K=4
p=1    
d=4
nb=500


for i in range(nb):
    
    name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(i)

        # tensors names
    sampletensor = 'samples' + name  # choose name of existing or to be created tensor
    labeltensor  = 'labels' + name   # choose name of existing or to be created tensor

    CreateEquidistantTensors(TENSOR_PATH,sampletensor,labeltensor,N,k=K,p=p,d=d)    
    
    

def row_pairwise_distances(x, y=None):
    
#        m=x.size()[0]
#        n=y.size()[0]
#        dist=torch.zeros(m,n)

#        for i in range(m):
#            for j in range(n):
#                dist[i,j]=torch.sum(torch.pow(x[i,:]-y[j,:],2)) 
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    dist = torch.pow(x_norm + y_norm - 2 * torch.mm(x, torch.transpose(y, 0, 1)),1/2)
    dist[dist != dist] = 0
    
    return dist


dataset='equidistant'

TENSOR_PATH = './datasets/' + dataset +  '/tensors/'

for i in range(10,11):
    
    name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(i)

        # tensors names
    sampletensor = 'samples' + name  # choose name of existing or to be created tensor
    labeltensor  = 'labels' + name   # choose name of existing or to be created tensor

    equi_loader ,test_loader, chan, inputsize, usecat = loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)
# 
# within                        
w1=row_pairwise_distances(torch.squeeze(equi_loader.dataset.tensors[0][0:2]))                  
                        
# within                        
w2=row_pairwise_distances(torch.squeeze(equi_loader.dataset.tensors[0][2:4])) 



# between                        
b=row_pairwise_distances(torch.squeeze(equi_loader.dataset.tensors[0][0:2]),\
                       torch.squeeze(equi_loader.dataset.tensors[0][2:4]))                  


print(w1)                        
print(w2)
print(b)

#%% rainbow


import loaddata
reload(loaddata)
from loaddata import loadDataTensor   
import scipy
from  scipy import ndimage    
     
import glob
import gc

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp,conv_out_size, squaresize,plotcp
        
import matplotlib.pyplot as plt

import numpy.matlib

import numpy as np
        
N=1
#k=3
#p=3
r=0
#name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
#outdim=4
#chan=1
#inputsize=[1,3]
#beforenet = STACKEDAUTOENCODER(outdim,chan,inputsize)
#beforenet.load_state_dict(torch.load(NET_PATH+'beforenet'+ name+'_netrun'  +str(i)))
##        beforenet.layers[0].encode[3]=nn.Dropout(p=0.9)
#classifier = type(beforenet)(outdim,chan,inputsize)
#classifier.load_state_dict(torch.load(NET_PATH+'classifier'+ name+'_netrun'  +str(i)))
##        classifier.layers[0].encode[3]=nn.Dropout(p=0.6) 
#beforenet.eval()
#classifier.eval()



difference=0.001
#k1=0
#k2=1
#k3=2
##measurelayer=2

ma=1

pts=5000
i=0

for k in range(1,2):

    if k==1:
        pstart=1
    else:
        pstart=1
        
    
    
    for p in range(1,2):
        
        name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
        outdim=5
        chan=1
        inputsize=[1,1]
        beforenet = STACKEDAUTOENCODER(outdim,chan,inputsize)
        i=0
        beforenet.load_state_dict(torch.load(NET_PATH+'beforenet'+ name+'_netrun'  +str(i)))
        #        beforenet.layers[0].encode[3]=nn.Dropout(p=0.9)
        classifier = type(beforenet)(outdim,chan,inputsize)
        classifier.load_state_dict(torch.load(NET_PATH+'classifier'+ name+'_netrun'  +str(i)))
        #        classifier.layers[0].encode[3]=nn.Dropout(p=0.6) 
        beforenet.eval()
        classifier.eval()

        net=torch.load(NET_PATH+'beforenet'+ name+'_netrun'  +str(i))

        for measurelayer in range(6):
            
            points = np.random.uniform(0,ma,[pts,N])
            for i in range(k):
                points[:,i]=np.arange(0,ma,ma/pts)

        
        #    points[:,k]=np.random.uniform(0,1-difference,5000)
            idx=np.argsort(points[:,0], axis=0)  # not necessary if already sorted "arange"
            points=points[idx]
            
            equidispoint=np.copy(points)
            for i in range(k):
                equidispoint[:,i]=equidispoint[:,i]+difference
        
            
            points=torch.tensor(points,dtype=torch.float)  
            points=torch.unsqueeze(points,dim=1)
            points=torch.unsqueeze(points,dim=1)
            
            equidispoint=torch.tensor(equidispoint,dtype=torch.float)  
            equidispoint=torch.unsqueeze(equidispoint,dim=1)
            equidispoint=torch.unsqueeze(equidispoint,dim=1)
            
            
            points=points.clone().detach()
            equidispoint=equidispoint.clone().detach()
            
            
            inpu,out = beforenet(points,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True)
            w=beforenet.layers[measurelayer].encode[0].weight.data.cpu()
            b=beforenet.layers[measurelayer].encode[0].bias.data.cpu()
            inpu=inpu.cpu()
            out=out.cpu()        
            pointsautoactivation=(out-b)/torch.norm(w,dim=None)
            
            
            
            inpu,out = classifier(points,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True)
            
            
            w=classifier.layers[measurelayer].encode[0].weight.data.cpu()
            b=classifier.layers[measurelayer].encode[0].bias.data.cpu()
            inpu=inpu.cpu()
            out=out.cpu()        
            pointsclassifier=(out-b)/torch.norm(w,dim=None)
            
            
            
            inpu,out = beforenet(equidispoint,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True)
            w=beforenet.layers[measurelayer].encode[0].weight.data.cpu()
            b=beforenet.layers[measurelayer].encode[0].bias.data.cpu()
            inpu=inpu.cpu()
            out=out.cpu()        
            equidispointautoactivation=(out-b)/torch.norm(w,dim=None)
            
            
            
            
            inpu,out = classifier(equidispoint,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True)
            w=classifier.layers[measurelayer].encode[0].weight.data.cpu()
            b=classifier.layers[measurelayer].encode[0].bias.data.cpu()
            inpu=inpu.cpu()
            out=out.cpu()        
            equidispointclassifier=(out-b)/torch.norm(w,dim=None)
            
            
            diffclass=equidispointclassifier-pointsclassifier
#            diffclass1= pointsclassifier
#            diffclass2= equidispointclassifier


            diffauto=equidispointautoactivation-pointsautoactivation
            
            
            
            spec=torch.norm(diffclass,dim=1).detach().numpy()
#            spec1=torch.norm(diffclass1,dim=1).detach().numpy()
#            spec2=torch.norm(diffclass2,dim=1).detach().numpy()

#            spec1=torch.norm(equidispointclassifier,dim=1).detach().numpy()
#            spec2=torch.norm(pointsclassifier,dim=1).detach().numpy()


#            spec=torch.norm(diffclass,dim=1).detach().numpy()
            
#            spec=scipy.ndimage.filters.gaussian_filter(spec, 40, order=0,output=None, mode='reflect', cval=0.0, truncate=12.0)
            
            #plt.plot(spec)
            fig, ax = plt.subplots()
            
#            plot=plt.imshow(np.matlib.repmat(-(spec1-np.mean(spec1)),1000,1))
            plt.plot(spec-np.mean(spec),linewidth=5)
#            plt.plot(spec1-np.mean(spec1),linewidth=5)
#            plt.plot(spec2-np.mean(spec2),linewidth=5)


#            plt.plot(spec2-np.mean(spec2))
        
        #    plot.axes.get_yaxis().set_visible(False)
            
        #    def format_func(value, tick_number):
        #        # find number of multiples of pi/2
        #        N = value
        #        if N == 0:
        #            return "0"
        #        elif N ==1666:
        #            return "0.33"
        #        elif N ==3333:
        #            return "0.66"
        #        elif N ==5000:
        #            return "1"
        #    
        #    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        #    ax.set_adjustable('box-forced')
            
            
#            ax.xaxis.set_major_locator(plt.FixedLocator(range(0,5000,500)))
#            ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.arange(0,1,3)))
#            plt.savefig(FIGURE_PATH  +'continum_layer' +str(measurelayer)+ name +'.png',bbox_inches = 'tight',
#            pad_inches = 0)
        
        #    plt.savefig('./datasets/continuous/savedfigures/continum_layer' +str(measurelayer)+'.png',bbox_inches = 'tight',
        #    pad_inches = 0)
    
#    
#continuum=np.arange(0,ma,1/pts)
#plot=plt.imshow(np.matlib.repmat(-(continuum-np.mean(continuum)),1000,1),cmap='rainbow')
#ax.xaxis.set_major_locator(plt.FixedLocator(range(0,5000,500)))
#ax.xaxis.set_major_formatter(plt.FixedFormatter(numpy.arange(0,1,3)))
#plt.savefig(FIGURE_PATH  +'continum' + '.png',bbox_inches = 'tight',
#pad_inches = 0)
# %%
plt.plot(pointsclassifier[0:5000,2].detach().numpy()-1*equidispointclassifier[0:5000,2].detach().numpy(),linewidth=2)
      
# %%      
plt.plot(pointsclassifier[0:5000,1].detach().numpy(),linewidth=2)
plt.plot(pointsclassifier[0:5000,1].detach().numpy(),linewidth=2) 
plt.plot(pointsclassifier[0:5000,3].detach().numpy(),linewidth=2)        
plt.plot(pointsclassifier[0:5000,4].detach().numpy(),linewidth=2)  
plt.plot(pointsclassifier[0:5000,0].detach().numpy(),linewidth=2) 
        
        
        
        
#%% 
#ALSO FOR GOOGLE COLAB
    
    
 # PUT CHUNCK TENSORS BACK TOGETHER


import glob

TENSOR_PATH = './datasets/michel/tensors/'


for run in range(0,1):

    for k in range(3,6):
        
        for p in range(k,k+1):
          
              name= '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)

              name=name + '_*'
              tensorlist=glob.glob(TENSOR_PATH + 'samples' + name )
              labellist=glob.glob(TENSOR_PATH + 'labels' + name )
              
              #tensorlist=list(np.asarray(tensorlist[0:int(len(tensorlist)/2)]))

              samples=torch.tensor([])
              labels=torch.tensor([])

              t=0  
              for tensor in tensorlist :

                    sampletensor = 'samples' + tensorlist[t][-len(name):]  
                    labeltensor  = 'labels' +  tensorlist[t][-len(name):]
                    tmpsamples= torch.load( TENSOR_PATH + sampletensor)
                    tmplabels= torch.load( TENSOR_PATH + labeltensor)
                    samples=torch.cat((samples,tmpsamples),dim=0)
                    labels=torch.cat((labels,tmplabels),dim=0)
                    del tmpsamples, tmplabels
                    
                    
              name= '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)                  
              torch.save(samples, TENSOR_PATH + 'samples' + name )
                      
# %% rainbow animation
              
              
import loaddata
reload(loaddata)
from loaddata import loadDataTensor       
     
import glob
import gc


import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import visualize_rainbow


r=0
NB_samecondi_runs=1
NB_ofnetwork_runs=1
N=7


dataset='continuous'

TENSOR_PATH = './datasets/' + dataset +  '/tensors/'


for run in range(0,NB_samecondi_runs):

    for k in range(1,2):
        if k==1:
            pstart=1
        else:
            pstart=1
        
        
        for p in range(k,k+1):
            
              
            name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
#            IMAGEPATH = './datasets/' + dataset + '/images/images' + name + '/' # change accordingly
#            TENSOR_PATH = './datasets/' + dataset +  '/tensors/'
            
            # tensors names
#            sampletensor = 'samples' + name  # choose name of existing or to be created tensor
#            labeltensor  = 'labels' + name   # choose name of existing or to be created tensor

           
#                name=name+'_?'
            tensorlist=glob.glob(TENSOR_PATH + 'samples' + name )
            labellist=glob.glob(TENSOR_PATH + 'labels' + name )

            
            for i in range(NB_ofnetwork_runs):
                       
                  t=0  
                  for tensor in tensorlist :

#                        sampletensor = 'samples' + tensorlist[t][-len(name):]  
#                        labeltensor  = 'labels' +  tensorlist[t][-len(name):] 
#                        train_loader ,test_loader, chan, inputsize, usecat = loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False)




                        fullname=name + '_netrun_'   + str(i)
#                        beforenetlist=torch.load(NET_PATH + 'beforenetlayerlist' + fullname+ '.pt' )
                        classifierlist=torch.load(NET_PATH + 'classifierlist' + fullname+ '.pt' )
                        
                        
                        layerid=5
                        ani=visualize_rainbow(classifierlist, layerid=layerid)
                        
                        
                        ani.save(FIGURE_PATH + 'rainbowanimation' + fullname + '.mp4')
                        
                        del classifierlist
                        
# %% check performance curves
                        
r=0
NB_samecondi_runs=1
NB_ofnetwork_runs=1
N=3
i=0

#dataset='runs_ID_2020-03-22 13:02:06'

#fig, ax = plt.subplots()
for run in range(0,NB_samecondi_runs):

    for k in range(1,2):
        if k==1:
            pstart=1
        else:
            pstart=2
        
        
        for p in range(k,k+1):
            
              
            name= '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(r)
            fullname=name + '_netrun_'   + str(i)
            netname='stimuliwizetestloss'
#            netname='stimuliwizetestaccuracy'
#            netname='classi_accuracy'
#            netname='classi_testloss'



                        
            curve=np.load(NET_PATH+netname+fullname+'.npy')  
            
            
#            trail=0
#            plt.plot(curve[trail],label=str(k))                   
#            ax.legend()

            
            
#for k in range(curve.shape[0]):
#
#    top20=numpy.argsort(curve[k],0)[-20:-1]
#    print(top20)


#records_array = curve.flatten()
top60 = numpy.argsort(curve,axis=1)[:,-60:-1]
#sorted_records_array = records_array[idx_sort]
vals, idx_start, count = numpy.unique(top60, return_counts=True,
                                return_index=True)

sort=numpy.argsort(count)[-1:0:-1]
rankedcount=count[sort]
print(rankedcount)
print(vals[sort])

# %%
nameslist=np.load('./datasets/'+'textures_michel'+'/info/' + 'nameslist1.npy')
nameslist[vals[sort][0:72]]


liste=[]

for i in range(len(vals[sort][0:72])):
    
    liste.append((nameslist[vals[sort][0:72]][i][60:75].strip('"\'')))
    
print(liste)  

#%% check linear separability and visualise data 2D : copy past and improved from a separate file
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import torch


from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
#dataset='sophie/PAPER_set2_black_insteadof_darkgrey_fixedUfixedposi_CAREFUL_WITHIN_BIAS_DIST_ZERO_PAIRS_no_eval_mode_validation_No_mean_accu'

dataset='textures_michel'


N=3
k=1
p=1
run=0

linsepmat=np.zeros((N-1,N-1))


for run in range(1):
    for k in range(1,2):
        
        if k==1:
            pstart=1
        else:
            pstart=1
            
        for p in range(k,k+1):
            
#            k=3
#            p=3
            name = '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)
            
            
#            A=np.load('./datasets/' + dataset+ '/images/images'+name+ '/cat0/A.npy')
#            B=np.load('./datasets/' + dataset+ '/images/images'+name+ '/cat1/B.npy')
#            C=np.load('./datasets/' + dataset+ '/images/images'+name+ '/cat2/C.npy')
#            D=np.load('./datasets/' + dataset+ '/images/images'+name+ '/cat3/D.npy')
#            A=np.load('A.npy')
#            B=np.load('B.npy')
            
            samples=torch.load('./datasets/' + dataset+ '/tensors/samples'+name)
            labels=torch.load('./datasets/' + dataset+ '/tensors/labels'+name)
#            
            x=samples.view(samples.size(0),-1).numpy()
            y=np.reshape(labels.view(labels.size(0),-1).numpy(),labels.size(0))



#            
#            labelA=np.zeros((np.shape(A)[0]))
#            labelB=np.ones((np.shape(B)[0]))
##            labelC=2*np.ones((np.shape(C)[0]))
##            labelD=3*np.ones((np.shape(D)[0]))
###            
#            x=np.concatenate((A,B,C,D),axis=0)
#            y=np.concatenate((labelA,labelB,labelC,labelD),axis=None)
#            
#            x=np.concatenate((A,B),axis=0)
#            y=np.concatenate((labelA,labelB),axis=None)
            
#            svm=OneVsRestClassifier(LinearSVC(random_state=0,C=1))
#            c=100
#            svm=OneVsOneClassifier(LinearSVC(random_state=0,C=c, max_iter=5000))
            c=100
            svm = SVC(C=c, kernel='linear', random_state=0)
            svm.fit(x, y)
            linsepmat[k-1,p-1]=svm.fit(x, y).score(x, y)

print(np.round(linsepmat,3))            

print(np.diag(linsepmat,k=-4))
# %%

y = svm.decision_function(x)
w_norm = np.linalg.norm(svm.coef_)
dist = y #/ w_norm

# %%

dist_ranked = np.argsort(abs(dist))
top=np.flip(dist_ranked)[0:72]
print(top)


# %%

zist=np.load('./datasets/'+'textures_michel'+'/info/' + 'nameslist1.npy')
#supovec_ind=svm.support_

#nameslist[supovec_ind][0][60:66]

liste=[]

for i in range(len(top)):
    
    liste.append((nameslist[top][i][60:75].strip('"\'')))
    
print(liste)    


# %% view 2D

# %% view 2D
from importlib import reload
import loaddata
reload(loaddata)
from loaddata import loadDataTensor       
import matplotlib.pyplot as plt
     
import glob
import gc

import matplotlib    
import matplotlib.animation as animation
from matplotlib import cm
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA


import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import visualize_layer
import cv2

dataset='textures_lucas'



N=7
k=1
p=1
run=0

linsepmat=np.zeros((N-1,N-1))

stiset=0

for run in range(1):
    for k in range(1,N):
        
        if k==1:
            pstart=1
        else:
            pstart=1
            
        for p in range(k,k+1):
            
            stiset=stiset+1
            
#            k=3
#            p=3
            name = '_N' + str(N)+ '_k' +str(k) +'_p'  +str(p) +'_run'  +str(run)
            X=torch.load('./datasets/' + dataset+ '/tensors/samples'+name)
            labels=torch.load('./datasets/' + dataset+ '/tensors/labels'+name)
            
            chan=1
            for i in range(chan):
                a=torch.min(X[:,i,:,:])
                b=torch.max(X[:,i,:,:])           
                X[:,i,:,:]=((X[:,i,:,:]-a)/(b-a))
                    
            
            
            pick=np.random.permutation(X.size()[0])
            X=X[pick]
            labels=labels[pick]
                
#            X=X[labels==0]    
            
           
            
            fullname=name + '_netrun_'   + str(run)
            beforenetlist=torch.load(NET_PATH + 'beforenetlayerlist' + fullname+ '.pt' )
            classifierlist=torch.load(NET_PATH + 'classifierlist' + fullname+ '.pt' )                 
##            
#            
##            if torch.cuda.is_available():
##                        image=image.cuda()
#                        
#            
            res50_conv.cpu()
            X=res50_conv(X) 
            X=X.permute(0,3,2,1)
#            
#    #            image=image.cpu()
#    #            label=label.cpu()    
#                
#                
            lab=labels                    
#            X=image.clone()
    
            net=classifierlist[-1]
#            layerid=1
#            net=beforenetlist[layerid][-1]
#        
##            if torch.cuda.is_available():
##                classifier.cuda()
##                    classifier.cpu()
#        
#                
            inpu,X = net(X,encode=True,decode=False,
                              encodelayer=layerid,decodelayer=0,linmap=False)
#            
#            
            w=net.layers[layerid].encode[0].weight.data #.cpu()
            b=net.layers[layerid].encode[0].bias.data #.cpu()
            X=(X-b)/torch.norm(w,dim=None)
    
    



#            X = X.view(X.size(0), -1) 
#            X=torch.mean(X,dim=1)
#            X = torch.squeeze(X)
            X = X.view(X.size(0), -1) 
#

            
            
            #fourier michel
#            title=('Michel cat2'+  ' fourrier' + str(stiset)) 
#
#            ax2  = plt.subplot(1,1,1)
#
#            sp = np.fft.fft(X.data.numpy())
#            freq = X.shape[1]
#            explained_variance = np.var(sp, axis=0)
#            explained_variance_ratio = explained_variance / np.sum(explained_variance)
#            show=3000
#            plt.plot(np.arange(show), explained_variance[0:show])
#            plt.text(0.5, 1.08, title,horizontalalignment='center',fontsize=16,transform = ax2.transAxes)
#         
#            plt.savefig(FIGURE_PATH + 'cat 2 fourrier_data_set=' + str(stiset) + '.png')
            
#
#            #fourier texture
#            title=('Textures cat1'+  ' fourrier' + str(stiset)) 
#            hW, hH = 299, 299
#            ax2  = plt.subplot(1,1,1)
#
#            sp = np.fft.fft2(X.data.numpy())
#            explained_variance = np.var(sp, axis=0)
#            explained_variance_ratio = explained_variance / np.sum(explained_variance)
#
#            
#            sp = np.fft.fftshift(explained_variance)
#            P = np.abs(sp)         
#            zoom=50
#            plt.imshow(20*np.log10(P),cmap='gray')#,vmin=np.min(P),vmax=np.max(P)+1); 
##            cv2.imshow('disp', P[hH-zoom:hH+zoom,hW-zoom:hW+zoom])
#                  
##            plt.imshow(P);
#
#
#
#         
#            plt.savefig(FIGURE_PATH + 'textures cat 1 fourrier_data_set=' + str(stiset) + '.png')




            
#            
            plt.clf()
            X=X.cpu()
            
            title=('Lucas'+  'mds classi  data set  ' + str(stiset)) 
            numb_dict = {1: 'A', 0: 'B'}
            N_components=2
#            
#            #        reduce = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#            #        seed = np.random.RandomState(seed=1)
            reduce = MDS(n_components=N_components,dissimilarity="euclidean",random_state=10)
#                    reduce = LocallyLinearEmbedding(n_components=2,random_state=10)
#            
#            reduce = PCA(n_components=N_components,random_state=10)
#            
#            
#            
            plot_only = 100
#            #        plot_only=np.sort(np.random.choice(X.size(0),300,replace=False))
#            low_dim_embs = reduce.fit_transform(X.data.numpy().astype(np.float64)[:plot_only, :])
            low_dim_embs = reduce.fit_transform(X.data.numpy().astype(np.float64))
#            component=reduce.components_
#            explained_variance = np.var(low_dim_embs, axis=0)
#            explained_variance_ratio = explained_variance / np.sum(explained_variance)
#
#            plt.bar(np.arange(N_components),explained_variance)
#            
#
#            
#            
#            
#            
#            
            lab = labels.numpy()[:plot_only]
###            
            pc1=1
            pc2=2
            xcoor, ycoor = low_dim_embs[:, pc1-1], low_dim_embs[:, pc2-1]
            
#            xcoor, ycoor = low_dim_embs[:, pc1-1], numpy.zeros(len(low_dim_embs))

##
##            
            ax2  = plt.subplot(1,1,1)
            for x, y, s in zip(xcoor, ycoor, lab):
                c = cm.rainbow(int(255*s/9)); 
            
                plt.text(x, y,numb_dict[s], backgroundcolor=c, fontsize=8,\
                              bbox=dict(facecolor=c, edgecolor=c, boxstyle='round'))
                
                plt.xlim(xcoor.min(), xcoor.max()); plt.ylim(ycoor.min(), ycoor.max());
            plt.text(0.5, 1.08, title,horizontalalignment='center',fontsize=16,transform = ax2.transAxes)
            
            plt.ylabel("Component " + str(pc2),fontsize=15)
            plt.xlabel("Component " + str(pc1),fontsize=15)



            plt.savefig(FIGURE_PATH + 'mds_classi_data_set' + str(stiset) + '.png')

#            c1=numpy.reshape(component[0],(299,299))
#            c2=numpy.reshape(component[1],(299,299))
#            c3=numpy.reshape(component[2],(299,299))

            
            
#            plt.subplot(1,2,1)
            
#            plt.imshow(c1, cmap='gray')     
#            plt.axis('off')
#            
            
#            plt.subplot(1,2,2)
#            
#            plt.imshow(c2, cmap='gray')       
#            plt.axis('off')
            
#            c4=numpy.min(c1)*numpy.ones((299,20))
#            comp=numpy.hstack((c1,c4,c2))
#            #          title=('Michel cat2'+  ' fourrier' + str(stiset)) 
#
#
#            plt.imsave(FIGURE_PATH + 'first_2_components_lucas_set' + str(stiset) + '.png',comp,cmap='gray')
            
            
#            plt.savefig(FIGURE_PATH + 'first_2 _components' + str(stiset) + '.png',cmap='gray')






# %%
print(numpy.dot(component[0],component[1]))
            
c1=numpy.reshape(component[0],(299,299))
c2=numpy.reshape(component[1],(299,299))
c3=numpy.reshape(component[2],(299,299))


plt.subplot(1,3,1)

plt.imshow(c1, cmap='gray')     
plt.axis('off')


plt.subplot(1,3,2)

plt.imshow(c2, cmap='gray')       
plt.axis('off')

plt.subplot(1,3,3 )

plt.imshow(c3, cmap='gray')       
plt.axis('off')

comp=numpy.hstack((c1,c2,c3))

#plt.imsave(FIGURE_PATH + 'first_components' + str(stiset) + '.png',comp)

            
# %% Fourier
            
            
import matplotlib.pyplot as plt
X = X.view(X.size(0), -1) 
sp = np.fft.fft(X[800:-1])
freq = X.shape[1]
explained_variance = np.var(sp, axis=0)
explained_variance_ratio = explained_variance / np.sum(explained_variance)
show=3000
plt.plot(np.arange(show), explained_variance[0:show])
            
#plt.savefig(FIGURE_PATH + 'fourrier_data_set=' + str(stiset) + '.png')
            
            
            
            

#%% Just CP : by reloading saved nets : raw, before, ater
res50_conv=[]            

        
import glob
import gc

import loaddata
reload(loaddata)
from loaddata import loadDataTensor

import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp_raw,conv_out_size, squaresize,plotcp

percent_use=[1, 1, 1,1,1] 
nb_of_tensors=[2,2, 1, 1,1]


N=3
NB_samecondi_runs=1
NB_ofnetwork_runs=1
measurelayer=4

#datasetorigin='continuous'
#datasetorigin='disjunctivevec_nonlinsep'
dataset='textures_michel'



#cptype='_equi'
#tensors= '/tensors'+cptype+'/'
tensors= '/tensors/'
avgname='raw_store_avg_distances_all_runs'+cptype


# To store avg distance for CP evolution on conjunctive/disjunctive case in a loop
#store_avg_distances=torch.zeros(N-1,N-1,4)
raw_store_avg_distances_all_runs=torch.zeros(NB_samecondi_runs,NB_ofnetwork_runs,N-1,N-1,6)
#store_avg_distances_all_runs=torch.load(DATA_PATH+ 'store_avg_distances_all_runs')

#store_avg_distances_all_runs=torch.tensor([])

r=0
for run in range(0,NB_ofnetwork_runs):

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
                
                
                  

                  
                  dists=torch.zeros(len(tensorlist),6)
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
                        rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin=\
                        computecp_raw(usecat,beforenet,classifier,res50_conv,loader,measurelayer=measurelayer)
                        dists[t,:]=torch.tensor([rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])     
                        t=t+1
                  rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin= \
                  torch.mean(dists,dim=0)
#                  print(dists)
                  
              
                  raw_store_avg_distances_all_runs[r,i,k-1,p-1,:]=\
                  torch.tensor([rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])
                  torch.save(raw_store_avg_distances_all_runs, DATA_PATH+ avgname)

                
            gc.collect()   
    r=r+1








#%% 
# plot CP with raw, before, after

import numpy as np
import matplotlib.pyplot as plt
import net_classes_and_functions
reload(net_classes_and_functions)
from net_classes_and_functions import STACKEDAUTOENCODER,trainauto,trainclassifier,computecp_raw,conv_out_size, squaresize,plotcp_raw

#cptype='_equi'
cptype='_regular'

avgname='raw_store_avg_distances_all_runs'+cptype #

raw_store_avg_distances_all_runs=torch.load(DATA_PATH+avgname)

# reshape into (netruns * condiruns , k, p, 6)
raw_store_avg_distances_all_runs=raw_store_avg_distances_all_runs.view(-1, *(raw_store_avg_distances_all_runs.size()[2:]))  


N=3

#GLOBALCP=torch.zeros(N-1,N-1)
#COMPRESSION=torch.zeros(N-1,N-1)
#SEPARATION=torch.zeros(N-1,N-1)
#INITIAL_SEPARATION=torch.zeros(N-1,N-1)
#BEFORE_BETWEEN=torch.zeros(N-1,N-1)
#AFTER_BETWEEN=torch.zeros(N-1,N-1)
#BEFORE_WITHIN=torch.zeros(N-1,N-1)
#AFTER_WITHIN=torch.zeros(N-1,N-1)

for k in range(0,N-1):
    
    for p in range(0,k+1):
      
        meanonruns=torch.mean(raw_store_avg_distances_all_runs[:,k,p,:],dim=0)

        #meanonruns=torch.mean(store_avg_distances_all_runs[:,:,k,p,:],dim=0)
        #mean=torch.mean(meanonruns[:,k,p,:],dim=0)
        rawbetween,rawwithin,autobetween,autowithin,classibetween,classiwithin = [*meanonruns]
#        sepa=classibetween-autobetween
#        compre=classiwithin-autowithin
        
#        GLOBALCP[k,p] = sepa-compre
#        COMPRESSION[k,p]=compre
#        SEPARATION[k,p]=sepa
#        BEFORE_BETWEEN[k,p]=autobetween
#        AFTER_BETWEEN[k,p]=classibetween
#        BEFORE_WITHIN[k,p]=autowithin
#        AFTER_WITHIN[k,p]=classiwithin
#        
#        INITIAL_SEPARATION[k,p] = autobetween-autowithin
        
        
        plt.close()
        plotcp_raw(rawbetween,rawwithin,autobetween,autowithin,classibetween,classiwithin)
        

        plt.savefig(FIGURE_PATH +'raw_cp'  +'_N=' + str(N) + '_K=' + str(k+1) + '_P=' + str(p+1) +cptype+ '.png')


        
        
#torch.save(GLOBALCP, DATA_PATH + 'GLOBALCP'+cptype)
#torch.save(COMPRESSION, DATA_PATH + 'COMPRESSION'+cptype)
#torch.save(SEPARATION,DATA_PATH + 'SEPARATION'+cptype)
#torch.save(INITIAL_SEPARATION,DATA_PATH + 'INITIAL_SEPARATION'+cptype)
#
#torch.save(BEFORE_BETWEEN, DATA_PATH + 'BEFORE_BETWEEN'+cptype)
#torch.save(AFTER_BETWEEN, DATA_PATH + 'AFTER_BETWEEN'+cptype)
#torch.save(BEFORE_WITHIN,DATA_PATH + 'BEFORE_WITHIN'+cptype)
#torch.save(AFTER_WITHIN,DATA_PATH + 'AFTER_WITHIN'+cptype)

