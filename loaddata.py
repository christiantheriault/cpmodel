import os
import torch
import torch.utils
import scipy
from  scipy import ndimage
#import cv2
from torchvision.transforms import transforms
from torchvision import datasets
#import torchaudio
import torch
import numpy
import numpy as np
from scipy.io.wavfile import read
#import sounddevice as sd
#import soundfile as sf
import numpy.matlib
from numpy import matlib
import matplotlib
import matplotlib.pyplot as plt


# %% This function loads data from NumPy array OR from PyTorch tensors

def loadDataTensor(TENSOR_PATH,sampletensor,labeltensor,numpyformat=False):

    # INPUTS REQUIREMENTS
    
    # 4D tensor containing your data
    # Format must be A x B x C x D
    # A = number of samples
    # B = channels (1 for gray images, 1 for sounds, 3 for color images)
    # C = sample first dimension (1 if sound)
    # D = sample second dimension
    
    # 1D tensor containing the labels
    # Format must be an array of integers of length A
    # A = number of samples
    
    # OUTPUT
    # Train and test data
    # Input channels
    # Input size
    # ID of categories used 
        

    if numpyformat:

        # Load  numpy tensor    
        samples= numpy.load(TENSOR_PATH + sampletensor + '.npy')     
        labels= numpy.load(TENSOR_PATH + labeltensor + '.npy') 
        
        # Change tensors to torch format
        samples=torch.tensor(samples)
        labels=torch.tensor(labels)
        labels=labels.type(dtype=torch.long)
        samples=torch.tensor(samples,dtype=torch.float)
        
        # Save your tensors
        torch.save(samples,TENSOR_PATH +sampletensor)
        torch.save(labels,TENSOR_PATH + labeltensor)    
        
        
       
    samples= torch.load( TENSOR_PATH + sampletensor)
    labels= torch.load( TENSOR_PATH + labeltensor)
    

    
    
    # Create dataloaders

    
    labels=labels.type(dtype=torch.long)
    
    cat=torch.unique(labels,sorted=True)
    
    # Categories that you want to use
    usecat=cat[[0,1]]
    
    
    inputsize=[samples.size(2),samples.size(3)]
    
    # Input channels
    chan=samples.size(1)
    
    
    
    # This places all pixel value between 0 and 1, treating channels (colors) separately
#    for i in range(chan):
#        a=torch.min(samples[:,i,:,:])
#        b=torch.max(samples[:,i,:,:])           
#        samples[:,i,:,:]=((samples[:,i,:,:]-a)/(b-a))*2-1  #*2-1
    
    
    
    train_batch_size=20
    
    
    # percentage of samples to use in total
    #
    
    usetotal=1 #  (1 means 100 % )
    
    # percentage of samples use for training. The rest is for testing.
    usefortrain=1
    
    
    
    Xtrain=torch.Tensor([])
    Ytrain=torch.Tensor([]).type(dtype=torch.long)
    
    
    Xtest=torch.Tensor([])
    Ytest=torch.Tensor([]).type(dtype=torch.long)
    
    
    
    for i in usecat:
        idx = torch.squeeze(torch.nonzero(labels==i))   
        keeptotal=int(numpy.round(usetotal*len(idx))) 
        keepfortrain=int(numpy.round(usefortrain*keeptotal))
        idx=idx[0:keeptotal]
#        idx=numpy.random.permutation(idx[0:keeptotal])
        idxtrain=idx[0:keepfortrain]
        idxtest=idx[keepfortrain:]
    
        Xtrain=torch.cat((Xtrain,samples[idxtrain]),dim=0)
        Ytrain=torch.cat((Ytrain,labels[idxtrain]),dim=0)
        
        Xtest=torch.cat((Xtest,samples[idxtest]),dim=0)
        Ytest=torch.cat((Ytest,labels[idxtest]),dim=0)
    
    
#    Xtest, indices =np.unique(Xtest.numpy(),return_index=True,axis=0)
#    Xtest=torch.tensor(Xtest)
#    Ytest=Ytest[indices]
    
    
    # Create datasets and data laoders
    traindataset=torch.utils.data.TensorDataset(Xtrain,Ytrain)
    testdataset=torch.utils.data.TensorDataset(Xtest,Ytest)
    
#    batch_size=len(traindataset)
    train_loader=torch.utils.data.DataLoader(dataset=traindataset, batch_size=train_batch_size, shuffle=True,drop_last=True)
#    fulltrain_loader=torch.utils.data.DataLoader(dataset=traindataset, batch_size=len(traindataset), shuffle=False,drop_last=True)
#    test_batchsize=10  #400
    test_batchsize=len(traindataset)
    test_loader=torch.utils.data.DataLoader(dataset=traindataset, batch_size=test_batchsize, shuffle=True,drop_last=True)


    return train_loader ,test_loader, chan, inputsize, usecat

# %% LOADING FROM IMAGE FOLDERS : DO THIS ONLY ONCE
# IMAGES WILL BE SAVED IN A TENSOR


def CreateImageTensors(IMAGE_PATH,TENSOR_PATH,sampletensor,labeltensor):

    # INPUTS REQUIREMENTS
    
    # Path to one folder containing a subfolder for each category
    # Path to one folder where torch tensors will be saved
    # Name of sample and label tensors 
    
    
    # OUTPUT
    # train and test data
    # input channels
    # input size
    # ID of categories used 
    
        
    # Choose image size (all image will be resized to this size)
    inputsize=[299,299]
    
    # 3 channels for colors, 1 for grayscale
    chan=3
    
    trans = transforms.Compose([
#        transforms.Resize(inputsize),
        transforms.Grayscale(num_output_channels=chan),
        transforms.ToTensor(),
        transforms.Normalize((0,),(0.5,))
        ])
    
        
    
    # Read folders
    dataset = datasets.ImageFolder(root=IMAGE_PATH, transform=trans)
    
    
    # Create samples and label tensors
    samples=torch.empty([len(dataset),*list(dataset[0][0].size())])
    labels=torch.empty(len(dataset))
    for i in range(len(dataset)):
                samples[i]=dataset[i][0]
                labels[i]=dataset[i][1]
    
    
#    # This places all pixel value between 0 and 1, treating channels (colors) separately
    for i in range(chan):
        samples[:,i,:,:]=(samples[:,i,:,:]-torch.min(samples[:,i,:,:]))/torch.max(samples[:,i,:,:])
                                                 
          
    torch.save(samples, TENSOR_PATH + sampletensor)
    torch.save(labels,  TENSOR_PATH + labeltensor)                 
    

# %% LOADING FROM SOUND FOLDERS : DO THIS ONLY ONCE
# SOUNDS WILL BE SAVED IN A TENSOR
def CreateSoundTensors(SOUND_PATH,TENSOR_PATH,sampletensor,labeltensor,k):

            
    folders = os.listdir(SOUND_PATH)
    
    # Path to where you save your tensors
    
    # Choose the sound size (all sounds will be downsampled to this size)
    # Sounds are smoothened and dowsampled 
              
# get number of files : to initialize tensors  
    nb_of_files=0
    for folder in folders:
        paths=os.listdir(SOUND_PATH + folder)
        nb_of_files=nb_of_files+int(len(paths))

    # Create samples and label tensors

    samples=torch.zeros(nb_of_files,1,1,22049)
    labels=torch.zeros(nb_of_files,)
    nameslist=[]

    j=0
    for folder in folders:    
        paths=os.listdir(SOUND_PATH + folder)
#        pick=np.random.permutation(int(len(paths)))
#        paths=list(np.asarray(paths)[pick])
        

        for sound in paths:        

            sample, rate = sf.read(SOUND_PATH + folder+'/'+sound, dtype='float32')
            sample=torch.tensor(sample)
            nameslist.append(SOUND_PATH + folder+'/'+sound)
            

#            samplingrate=sample.size(0)/inputsize[1]
#            sigma=samplingrate/2
#            sample=scipy.ndimage.filters.gaussian_filter(sample.numpy(), 2, order=0,
#                                                   output=None, mode='reflect', cval=0.0, truncate=4.0)
#            sample=torch.tensor(sample)
#            sample=sample[torch.ceil(torch.linspace(0,sample.size(0)-1,inputsize[1])).type(dtype=torch.long)]
#            sample=torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(sample,dim=0),dim=0),dim=0)
            sample=torch.unsqueeze(torch.unsqueeze(sample,dim=0),dim=0)
            samples[j,:,:,:]=sample

            label=torch.tensor([folders.index(folder)],dtype=torch.float)
            #labels=torch.cat((labels,label),dim=0)
            labels[j]=label
            del sample, label
            j=j+1
            print(j)



    # This places all data point value between 0 and 1
    samples=(samples-torch.min(samples))/(torch.max(samples)-torch.min(samples))

    torch.save(samples, TENSOR_PATH + sampletensor)
    torch.save(labels, TENSOR_PATH + labeltensor)
    np.save('./datasets/'+'textures_michel'+'/info/' + 'nameslist' + str(k), nameslist)




# %% CREATE MY CONTINUOUS STIMULI FOR CONTINUUM HYPOTHESIS : RAINBOW EFFECT

def CreateContinuumTensors(TENSOR_PATH,sampletensor,labeltensor,interval,boundaries,N,k,nb):
    
    samples = numpy.empty((0,N))
    labels = numpy.empty((0))
#    interval=[0,1]
#    boundaries=[0, 0.33, 0.66, 1]
#    k=0
    
    for i in range(len(boundaries)-1):
#        cat= numpy.random.uniform(0.2,0.3,[nb,N])
#        cat= numpy.random.uniform(interval[0],interval[1],[nb,N])
        cat= 0.5*np.ones((nb,N))
        cat[:,0]=numpy.random.uniform(boundaries[i],boundaries[i+1],[nb])
#        cat[:,1]=numpy.random.uniform(boundaries[i],boundaries[i+1],[nb])
#        cat[:,2]=numpy.random.uniform(boundaries[i],boundaries[i+1],[nb])
        samples=numpy.concatenate((samples,cat))
        label=i*numpy.ones((nb))
        labels=numpy.concatenate((labels,label))
        
        
    
    samples=torch.tensor(samples,dtype=torch.float)  
    samples=torch.unsqueeze(samples,dim=1)
    samples=torch.unsqueeze(samples,dim=1)
    labels=torch.tensor(labels,dtype=torch.long)
    
    r=torch.randperm(labels.size(0))
    
    labels=labels[r]
    samples=samples[r]
    
    
        
    
    
    # Save your tensors
    torch.save(samples,TENSOR_PATH +sampletensor)
    torch.save(labels,TENSOR_PATH + labeltensor) 
    
    


# %% CREATE SOPHIE's STIMULI in non visual mode

def CreateSophieTensors(TENSOR_PATH,sampletensor,labeltensor,N,k,p,nb):
    
#    k=2
#    p=2
#    nb = 200
#    N = 8

#    samples=torch.zeros(2*nb,1,32,32)
#    labels=torch.zeros(2*nb)

    
    #U
    
    U = np.zeros([3,N])
    
    for i in range(0,N):
        U[:,i] = np.random.permutation([0,-1,1])
    
    a = U[0,:]
    b = U[1,:]
    n = U[2,:]
    
    
#    order  = np.random.permutation(N) #ordre utiliser pour rendre aléatoire la position de chaque attribut 
    order = np.arange(N)
    #disjonction inclusive (au moins p)
    
    
    #Générer les stimuli faisant parti de la catégorie A
    A_inclu = np.zeros([nb,N])
    
    nbStim = 0
    while nbStim < nb:
        for i in range(p,k+1): #prendre toutes les combinaisons de "au moins p"
            comb = np.random.choice(k,i,replace=False)
            #print(comb)
            comb = np.sort(comb)
        
            placeInComb = 0
            countB = 0 
            x = np.zeros(N)
            for j in range(0,k): #générer les k dimensions
                if j == comb[placeInComb]:
                    x[j] = a[j] #mettre valeur associée à A dans les positions générées par comb
                    if placeInComb < i-1:
                        placeInComb = placeInComb +1
                else:
                    BorN = np.random.randint(0,2)
                    if BorN == 0 and countB < p-1: #permettre <p dimensions prenant la valeur de B
                        x[j] = b[j]
                        countB = countB+1
                    else:
                        x[j] = n[j]
                
            for j in range(k,N): #ajouter le bruit pour les dimensions restantes 
                x[j]= U[np.random.randint(0,3)][j]
            
            if nbStim < nb:
                A_inclu[nbStim] = x
                nbStim = nbStim + 1   
    
    for i in range(0,nb): #changer la position de chaque attribut 
        A_inclu[i] = A_inclu[i][order]
    
    
    
    #Générer les stimuli faisant parti de la catégorie B
    B_inclu = np.zeros([nb,N])
    
    nbStim = 0
    while nbStim < nb:
        for i in range(p,k+1): #prendre toutes les combinaisons de "au moins p"
            comb = np.random.choice(k,i,replace=False)
            comb = np.sort(comb)
        
            placeInComb = 0
            countA = 0 
            x = np.zeros(N)
            for j in range(0,k): #générer les k dimensions
                if j == comb[placeInComb]:
                    x[j] = b[j] #mettre valeur associée à B dans les positions générées par comb
                    if placeInComb < i-1:
                        placeInComb = placeInComb +1
                else:
                    AorN = np.random.randint(0,2)
                    if AorN == 0 and countA < p-1: #permettre <p dimensions prenant la valeur de A
                        x[j] = a[j]
                        countA = countA+1
                    else:
                        x[j] = n[j]
                
            for j in range(k,N): #ajouter le bruit pour les dimensions restantes 
                x[j]= U[np.random.randint(0,3)][j]                    
            
            if nbStim < nb:
                B_inclu[nbStim] = x
                nbStim = nbStim + 1  
                
    for i in range(0,nb): #changer la position de chaque attribut 
        B_inclu[i] = B_inclu[i][order]
        
        
    B_inclu=torch.tensor(B_inclu,dtype=torch.float)
    A_inclu=torch.tensor(A_inclu,dtype=torch.float)

    
    return A_inclu,B_inclu,U
    
    
    

# %% Generate 2 dijunctive categories of abstract vectors

def generate_disjunctive_2cat(nb, N , k ,p):
    
    
       # generate two disjunctive categories : conjunctive is when p = k    
    
    
    
       # nb : number of stimuli per category
       # N : number of dimensions
       # k : number of category relevant dimensions
       # p : minimal number of "correct value" dimensions among the k relevant dimensions       
        
    
       # all possible values at each k relevant dimensions
        
        # First  row  of  U is for cat A
        # Second row  of  U is for cat B
        # Third  row  of  U is neither   MAY NEVER USE THIRD ROW (SEE BELOW) IF WANT a SIMPLE A VS NOT A
        # It suffice not use p=1 for k > 1 (not need for third row)
        
        u=[1,-1,0]
#        u=[1,0,-1]    # non linearly separable
        
#        u=[1,-1]
        U = np.zeros([3,N])
#        U = np.zeros([2,N])

        for i in range(0,N):
#            U[:,i] = np.random.permutation(u)
            U[:,i] = u

            
#        U=np.asarray([np.ones(N),-1*np.ones(N),0*np.ones(N)])    # may not use third row (optional see below)
#        U=np.asarray([np.ones(N),-1*np.ones(N)])    # may not use third row (optional see below)

    
    #    choose which k dimensions are covariant
        covariant_positions=np.random.choice(np.arange(N),k,replace=False) # choose id of covariants
#        covariant_positions=np.arange(k)
        # rest of dimensions are irrelevant
        not_relevant_positions=np.array(list(set(np.arange(N))-set(covariant_positions)))  
        
        
    #   CAT A
    #   set values of relevant dimensions (first row of U)
        K= np.matlib.repmat(U[0,covariant_positions],nb,1)
        
    
        # flip at most K-p values, to make sure you have at least p not flipped. Choose new value randomly from rows 2 and 3(if joker) of U 
        # make sure that less than than p values are fliped to the other category (preventing membership to 2 categories)
        
        for i in range(nb):
#            numb_of_flip=int(np.random.randint(0,np.shape(K)[1]-p+1,(1,)))
            numb_of_flip=int(np.shape(K)[1]-p) # exclusive disjunctive exaclty p
            randpo=np.random.choice(np.arange(k),numb_of_flip,replace=False)
            notreached=True
            while notreached :      # make sure that less than p values are flipped to other category
                flipval=U[np.random.choice([1,2],(1,numb_of_flip)),randpo]  # third row not used if just [1] third row is used if [1,2]
                if np.count_nonzero(flipval-U[1,randpo]==0)<p:     # make sure than less than p are flipped to other cat
                   K[i,randpo]=flipval
                   notreached = False
                
        # irrelevant dimensions in U use [0:3] if joker
        n=np.random.choice(u[0:3],(nb,N-k))
            
        # put together relevant and irrelevant dimensions
        A=np.zeros((nb,N))
        A[:,covariant_positions]=K
        if len(not_relevant_positions)!=0:
            A[:,not_relevant_positions]=n 
    
            
    #   CAT B
    #   set values of relevant dimensions (second row of U)
        L= np.matlib.repmat(U[1,covariant_positions],nb,1)
#        L= np.matlib.repmat(U[0,covariant_positions],nb,1)# for not A instead of B
        
        # flip at most K-p values, to make sure you have at least p not flipped. Choose new value randomly from rows 2 and 3(if joker) of U 
        # make sure that less than than p values are fliped to the other category (preventing membership to 2 categories)
        
        for i in range(nb):
#            numb_of_flip=int(np.random.randint(0,np.shape(L)[1]-p+1,(1,)))
            numb_of_flip=int(np.shape(L)[1]-p) # exclusive dijunctive exactly p 
#            numb_of_flip=int(np.random.randint(np.shape(L)[1]-p+1,np.shape(L)[1]+1,(1,))) # for not A instead of B
            randpo=np.random.choice(np.arange(k),numb_of_flip,replace=False)
            notreached=True
            while notreached :      # make sure that less than p values are flipped to other category
                flipval=U[np.random.choice([0,2],(1,numb_of_flip)),randpo]  # third row not used if just [0]  third row is used if [0,2]
#                flipval=U[np.random.choice([1,2],(1,numb_of_flip)),randpo]  # for not A instead of B
                if np.count_nonzero(flipval-U[0,randpo]==0)<p:
#                if np.count_nonzero(flipval-U[0,randpo]==0)==0:       # not A instead of B : useless because of line above (just not to change code)
                   L[i,randpo]=flipval
                   notreached = False
        # irrelevant dimensions in U use [0:3] if joker
        n=np.random.choice(u[0:3],(nb,N-k))
            
        # put together relevant and irrelevant dimensions
        B=np.zeros((nb,N))
        B[:,covariant_positions]=L
        if len(not_relevant_positions)!=0:
            B[:,not_relevant_positions]=n
            
    
        

            
        return A,B,U,covariant_positions

# %% Generate 2 dijunctive categories of abstract vectors : this version can't use p=1, must start at p=2. No joker value

def generate_disjunctive_cat(numbofcats,nb, N , k ,p):
        
       # generate disjunctive categories : conjunctive is when p = k    
    
       # nb : number of stimuli per category
       # N : number of dimensions
       # k : number of category relevant dimensions
       # p : minimal number of "correct value" dimensions among the k relevant dimensions       
        
        # matrix U
        # all possible values at each k relevant dimensions
        # First  row  of  U is for cat 1
        # Second row  of  U is for cat 2
        # etc.
        # last  row  of  U is neither : value is 0
        
#        numbofcats=4
        u=np.linspace(1,-1,numbofcats)
        if (numbofcats)%2==1:
            u=np.delete(u,int((len(u)-1)/2))  # if odd delete the 0 in the middle : -1 because begins at 0
            u=np.concatenate((u,[0]))
        else:
            u=np.concatenate((u,[0]))
            
        
        
#        u=u[0:-1]  # if you don't the last boundary  : just the left boundaries for continuum
#        
#        if numbofcats % 2 == 0 :
#            u=np.concatenate((np.linspace(0,1,numbofcats),np.asarray([0])),axis=0)
#        else :
#            u=np.linspace(0,1,numbofcats)    
#        
#        u=np.transpose(np.asarray([[1,0.5,0.75]]),axes=[1,0])
    
        U = np.zeros([np.shape(u)[0],N])
        for i in range(N):
            U[:,i] = np.squeeze(u)
#            U[:,i] = np.random.permutation(np.squeeze(u))
#        U=np.matlib.repmat(u,1,N)
#        numcats=np.shape(U)[0]-1
        numcats=np.shape(U)[0]-1
        cats=[]
        
    
    #    choose which k dimensions are covariant
#        covariant_positions=np.random.choice(np.arange(N),k,replace=False) # choose id of covariants
        covariant_positions=np.arange(k)
        # rest of dimensions are irrelevant
        not_relevant_positions=np.array(list(set(np.arange(N))-set(covariant_positions)))  
        
        
    #   CATS 
    
        for j in range(numcats):   # keep last row for "anything category"
            
                K= np.matlib.repmat(U[j,covariant_positions],nb,1)

            
                # flip at most K-p values. Choose new value randomly from other rows of U 
                # make sure that less than than p values are fliped to another category (preventing membership to 2 categories)
                
                for i in range(nb):
                    numb_of_flip=int(np.random.randint(0,np.shape(K)[1]-p+1,(1,)))
                    randpo=np.random.choice(np.arange(k),numb_of_flip,replace=False)
                    notreached=True
                    while notreached :      # make sure that less than p values are flipped to other category
#                        flipfrom=np.array(list(set(np.arange(numcats+1))-set(np.array([j])))) # don't pick for current category
                        flipfrom=np.array(list(set(np.arange(numcats+1))-set(np.array([j])))) # don't pick for current category
                        flipval=U[np.random.choice(flipfrom,(1,numb_of_flip)),randpo]  # select available values in U
                        riskyflips=np.array(list(set(np.arange(numcats))-set(np.array([j])))) # everythng exept last row of U and the current category
                        countriskyflipts=[]
                        for s in range(len(riskyflips)):
                            countriskyflipts.append(np.count_nonzero(flipval-U[riskyflips[s],randpo]==0)) # check number of flips to other cat for all cat                            
#                        print('countriskyflipts = ' + str(countriskyflipts)+ ' for k=' + str(k) +' and p= '+ str(p) )
                        if max(countriskyflipts)<p:   # if no cat were flipped to p times, then its a valid flip, stop
                           K[i,randpo]=flipval
                           notreached = False
                        
                # irrelevant dimensions : random values in U
                n=np.random.choice(np.squeeze(u),(nb,N-k))
                    
                # put together relevant and irrelevant dimensions
                A=np.zeros((nb,N))
                A[:,covariant_positions]=K
                if len(not_relevant_positions)!=0:
                    A[:,not_relevant_positions]=n 
            
    
                cats.append(A)

            
    
        

            
        return cats,U,covariant_positions

# %% Create disjunctive textures (KL) from dijunctive abstract vectors : 
#    textures are  numpy arrays, not images 
    
def generate_KL_arrays(A,B,U,imagesizeX, imagesizeY, microsizeX , microsizeY):

#   microsizeX  : horizontal size of micro features 
#   microsizeY  : vertical size of micro features     
#   N : number of micro features
#   imagesizeX  : horizontal size of image
#   imagesizeY : vertical size of image
#   k : number of category covariants
#   nb : number of sitmuli per category 
    
    
    if not np.remainder(imagesizeY,microsizeY)==0 :
            raise ValueError('Vectical image size must be dividable by micro feature verctical size')
    
    if not  np.remainder(imagesizeX,microsizeX)==0:
            raise ValueError('Horizontal image size must be dividable by micro feature horizontal size')
            
    
    nb=np.shape(A)[0]
    N= np.shape(A)[1] 
    # stimuli container for categories
    vectorstimuli=np.zeros((2,nb , N))
#    arraysstimuli=np.zeros((2,nb,imagesizeX,imagesizeX))
    arraysstimuli=np.zeros((2,nb,imagesizeX,imagesizeX,3))  # for rgb


   # vector stimuli to be translated into pixel stimuli

    vectorstimuli[0]=A
    vectorstimuli[1]=B


    # create micro feautres (pixels)
    # make sure all are different
    
    micro_ids=np.arange(N)
    all_micros=np.random.rand(3*N,microsizeY,microsizeX)
    
    # make sure balanced number of 0 (blcck) and 1 (white)
    nb_ones=int(np.floor(microsizeY*microsizeY/2))
    nb_zeros=int(np.ceil(microsizeY*microsizeY/2))
    create_micro_from_this=np.concatenate((np.ones((1,nb_ones)),np.zeros((1,nb_zeros))),axis=1)

    for i in range(3*N):
            keep_searching=True
            while keep_searching:
                    micro=np.reshape(np.random.permutation(*create_micro_from_this),(microsizeY,microsizeX)) # this the regular black and white version
####                    micro=np.random.randint(0,2,(microsizeY,microsizeX)
#                    micro=np.random.uniform(0,1,(microsizeY,microsizeX))   # this is the "continuous gray scale" version
                    if 0 not in np.sum(abs(all_micros-micro),(1,2)):
                        keep_searching=False
            all_micros[i]=micro
            
            

    # organizse micro features as binaries : stack in two rows
    all_microsbinary=np.random.rand(3,N,microsizeY,microsizeX)
    all_microsbinary[0]=all_micros[0:N]
    all_microsbinary[1]=all_micros[N:2*N]
    all_microsbinary[2]=all_micros[2*N:]


    
    
    
    # translate vector stimuli into pixels
    
    # loop on categories
    for catid in range(np.shape(vectorstimuli)[0]):
        
        # loop on images
        for im in range(nb):
    
            # create a matrix filled with numbers representing each micro feature
            # all these numbers wiil be replace by the corresponding pixels micro features
            mat=np.random.rand(int(imagesizeY/microsizeY),int(imagesizeX/microsizeX))
            a=np.shape(mat)[0]
            b=np.shape(mat)[1]
            
            # make sure that no feature id is consecutive verticaly and horizontaly
            for i in range(a):
                for j in range(b):
#                 neighborhood=[ mat[max(i-1,0),j] , mat[min(i+1,a-1),j]  ,mat[i,max(j-1,0)] , mat[i, min(j+1,b-1)]  ]
                 neighborhood=mat[max(i-1,0) : min(i+1,a-1)+1  , max(j-1,0) : min(j+1,b-1)+1  ].flatten()
                 possible=list(set(micro_ids)-set(neighborhood))
                 if len(possible)==0:
                      print('reached')
                      possible=micro_ids
              
                 mat[i,j]= int(possible[np.random.randint(len(possible))])
                
            # fill in the canvas with pixels
#            image=np.zeros((imagesizeY,imagesizeX))
            image=np.zeros((imagesizeY,imagesizeX,3))  # fpr rgb

            for i in range(a):             
                for j in range(b):
                    micro_id=int(mat[i,j])
                    dim=U[:,micro_id]
                    row=dim[np.where(dim==int(vectorstimuli[catid,im,micro_id]))]
#                    image[i*microsizeY:(i+1)*microsizeY,j*microsizeX:(j+1)*microsizeX]=\            
                    image[i*microsizeY:(i+1)*microsizeY,j*microsizeX:(j+1)*microsizeX,0]=\
                    all_microsbinary[int(row)][micro_id]    # for rgb one row above
        
        
            arraysstimuli[catid,im,:,:]=image  
    
    return arraysstimuli, all_microsbinary


# %%


def CreateTensorsFrom1Darrays(IMAGEPATH,TENSOR_PATH,sampletensor,labeltensor):

    
    
    folders = sorted(os.listdir(IMAGEPATH))

    samples=torch.tensor([])
    labels=torch.tensor([])
        
    for folder in folders:
        paths=sorted(os.listdir(IMAGEPATH + folder))
        for mat in paths:            
            sample=torch.tensor(np.load(IMAGEPATH + folder+'/'+ mat))
            sample=torch.unsqueeze(sample,dim=1)
            sample=torch.unsqueeze(sample,dim=1)
            sample=torch.tensor(sample.clone().detach(),dtype=torch.float)  
            samples=torch.cat((samples,sample),dim=0)
            labels=torch.cat((labels,folders.index(folder)*torch.ones(sample.size(0))),dim=0)
            
            
        
#    r=torch.randperm(samples.size(0))
#    samples=samples[r,:,:,:]
#    labels=labels[r]
    
    
    # Save your tensors
    torch.save(samples,TENSOR_PATH +sampletensor)
    torch.save(labels,TENSOR_PATH + labeltensor) 
    
# %%


def CreateTensorsFrom2Darrays(IMAGEPATH,TENSOR_PATH,sampletensor,labeltensor):

    
    
    folders = os.listdir(IMAGEPATH)

    samples=torch.tensor([])
    labels=torch.tensor([])
        
    for folder in folders:
        paths=os.listdir(IMAGEPATH + folder)
        for mat in paths:            
            sample=torch.tensor(np.load(IMAGEPATH + folder+'/'+ mat))
            sample=torch.unsqueeze(sample,dim=1)
            sample=torch.tensor(sample.clone().detach(),dtype=torch.float)  
            samples=torch.cat((samples,sample),dim=0)
            labels=torch.cat((labels,folders.index(folder)*torch.ones(sample.size(0))),dim=0)
            
            
        
    r=torch.randperm(samples.size(0))
    samples=samples[r,:,:,:]
    labels=labels[r]
    
    
    # Save your tensors
    torch.save(samples,TENSOR_PATH +sampletensor)
    torch.save(labels,TENSOR_PATH + labeltensor) 
        
    

# %% CREATE equidisance quatuor STIMULI


#  IN PROGRESS


#def GenerateEquidistantQuatuors(U,covariant_positions,N,k,p,d):
def GenerateEquidistantQuatuors(N,k,p,d):
    
    # for now this function takes as relevant dimension the first k dimesion (not random)
    # it does not matter for its purpose 
    
    # moreover, the flip for disjunctive must flipped to 0 never to the other category
    # and the irrelevant does not contain 0
    
    # all possible values at each k relevant dimensions
    
    # First  row  of  U is for cat A
    # Second row  of  U is for cat B
    # Third  row  of  U is neither

    
#    u=[1,-1,0]
    u=[1,0,-1]
    
    
    
#    u=[0,-1,1]

    U = np.zeros([3,N])
    for i in range(0,N):
        U[:,i] = u   # changed from original disjunctive function
#        U[:,i] = np.random.permutation(u)
#        
##    choose which k dimensions are covariant
##    covariant_positions=np.arange(k)
    covariant_positions=np.random.choice(np.arange(N),k,replace=False) # choose id of covariants

    # rest of dimensions are irrelevant
    not_relevant_positions=np.array(list(set(np.arange(N))-set(covariant_positions)))  

    nb=1  # only need one   # changed from original disjunctive function

#   CAT A

#   set values of relevant dimensions (first row of U)
#    K= np.matlib.repmat(U[0,:],nb,1)
    K= np.matlib.repmat(U[0,covariant_positions],nb,1)

    
    
    # for disjunctive case
    # flip at most K-p values. Choose new value randomly from bottom rows of U (from noise or from other category)
    # make sure that less than than p values are fliped to the other category (preventing membership to 2 categories)
    
    for i in range(nb):
#        numb_of_flip=int(np.random.randint(0,np.shape(K)[1]-p+1,(1,)))
        numb_of_flip=int(np.shape(K)[1]-p) # exclusive disjunctive exaclty p

        randpo=np.random.choice(np.arange(k),numb_of_flip,replace=False)
        notreached=True
        while notreached :      # make sure that less than p values are flipped to other category
            flipval=U[np.random.choice([1,2],(1,numb_of_flip)),randpo]  # changed from original disjunctive function
            if np.count_nonzero(flipval-U[1,randpo]==0)<p:
               K[i,randpo]=flipval
               notreached = False
    # irrelevant dimensions
    
    # trick to deal with the random aspect of disjunctive : repeat K in noise part such that the within flips below match between flip
    # this works when k=d
    n=np.concatenate((K,np.random.choice(u,(nb,N-2*k))),axis=1)
#    n=np.random.choice(u[0:2],(nb,N-k))   # changed from original disjunctive function
        
    # put together relevent and irrelevant dimension
    A1=np.zeros((nb,N))
    A1[:,covariant_positions]=K
    if len(not_relevant_positions)!=0:
        A1[:,not_relevant_positions]=n 
 

   
        
    # put in torch format    
    A1=torch.tensor(A1).type(dtype=torch.FloatTensor)
    
    
##    previous code (not quite disjunctive)
    # relevant dimensions
#    K = torch.ones(1,k,dtype=torch.long)
#    randpo=torch.randperm(K.size(1))
#    numb_of_flip=K.size(1)-p
#    K[0,randpo[0:numb_of_flip]]=torch.zeros(1,numb_of_flip,dtype=torch.long)
#    # irrelevant dimensions
#    n=torch.randint(0, 2, (1,N-k),dtype=torch.long)
#    n[n==0]=-1
#    A1 = torch.cat((K, n),dim=1)
    
    
    
    
    A1=A1.type(torch.FloatTensor)
    A2=A1.clone()
    
#   flip d elements in the random port of stimulus (in the N-k ranndom elements)
#    randpo=np.random.permutation(not_relevant_positions)
    randpo=torch.arange(N-k)+k  # flip the first k 
    
    for i in range(d):
        if A2[0,not_relevant_positions[i]]==1:
                A2[0,not_relevant_positions[i]]=-1
        else:
                if A2[0,not_relevant_positions[i]]==-1:
                    A2[0,not_relevant_positions[i]]=1     
    
    

#   position of unchanced in the random
    unchanged=torch.from_numpy(numpy.setdiff1d(numpy.arange(k,N),randpo[0:d].numpy()))


#  create the same pair in the other category

    B1=A1.clone()
    B2=A2.clone()
    
#    B1[0,covariant_position]=-1
#    B2[0,covariant_position]=-1
    
    for i in range(k):
        if B1[0,covariant_positions[i]]==1:
                B1[0,covariant_positions[i]]=-1
        else:
                if B1[0,covariant_positions[i]]==-1:
                    B1[0,covariant_positions[i]]=1   
        
        if B2[0,covariant_positions[i]]==1:
                B2[0,covariant_positions[i]]=-1
        else:
                if B2[0,covariant_positions[i]]==-1:
                    B2[0,covariant_positions[i]]=1
    
   
    
#   adjust between distance by fliping d-k element in the unchanged
    randpo=torch.randperm(len(unchanged))
    numb_of_flip=d-k
    for i in range(numb_of_flip):
        
        
        if B1[0,unchanged[randpo[i]]]==1:
                B1[0,unchanged[randpo[i]]]=-1
                B2[0,unchanged[randpo[i]]]=-1
        else:
                if B1[0,unchanged[randpo[i]]]==-1:
                    B1[0,unchanged[randpo[i]]]=1 
                    B2[0,unchanged[randpo[i]]]=1 
                
    
#        if B1[0,unchanged[randpo[i]]]==1 :
#              B1[0,unchanged[randpo[i]]]=-1
#              B2[0,unchanged[randpo[i]]]=-1
#        else:
#              B1[0,unchanged[randpo[i]]]=1
#              B2[0,unchanged[randpo[i]]]=1

            
            
            
    
    A=torch.cat((A1, A2),dim=0)
    B=torch.cat((B1, B2),dim=0)
    
    

#    samples=torch.cat((A, B),dim=0)
##    r=torch.randperm(samples.size(1))
##    samples=samples[:,r]
#    samples=torch.unsqueeze(samples,dim=1)
#    samples=torch.unsqueeze(samples,dim=1)
#    
#        
#    labels = torch.cat(( torch.flatten(torch.zeros([1 ,2],dtype=torch.long)[0,:]),
#                                          torch.flatten(torch.ones([1 ,2],dtype=torch.long)[0,:])      
#                                          ))   
##    r=torch.randperm(labels.size(0))
##    
##    labels=labels[r]
##    samples=samples[r,:]
    
    
    
    
#    # Save your tensors
#    torch.save(samples,TENSOR_PATH +sampletensor)
#    torch.save(labels,TENSOR_PATH + labeltensor) 
    
    return A,B,U,covariant_positions
    
    