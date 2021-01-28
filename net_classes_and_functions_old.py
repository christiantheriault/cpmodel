import torch
import torch.nn as nn
import torch.utils
from torch.autograd import Variable
import numpy as np
import math
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import copy
import gc
from scipy.spatial import distance_matrix



# %%  fully connected autoencoder class

# IMPUT : inputsize, outputsize
# OUTPUT : output activation 


class AUTOENCODER(nn.Module):
    
    def __init__(self,insize,innersize,top=False):
        
        super(AUTOENCODER, self).__init__()
    
    
        if top==True:
            self.encode=nn.Sequential(
            nn.Linear(insize , innersize),
            nn.Sigmoid()
            )
        else :
            
            self.encode=nn.Sequential(
            nn.Linear(insize , innersize),
            nn.BatchNorm1d(innersize,insize),
            nn.LeakyReLU(1), #
#            nn.Sigmoid(),
#            nn.Tanh(),
            nn.Dropout(p=0.0)
            )
                
        self.decode=nn.Sequential(      
        nn.Linear(innersize, insize),
        nn.BatchNorm1d(insize,innersize)
#        nn.Tanh()
        )
#        nn.Tanh()
        
    def forward(self, x,encode=True,decode=True,noise=False):
        
        if noise :
            noise=2*(torch.rand(x.size()))-1
            if torch.cuda.is_available():
                noise=noise.cuda()
            x = torch.add(x, 0.1 * (noise))

        
        if encode :
            x=self.encode(x)
            
        if decode:
            x=self.decode(x)
                
        return x


# %%    CONVOLUTIONAL AUTOENCODING
class CONVAUTOENCODER(nn.Module):
    
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,poolsize, poolstride,poolpadding):
        
        super(CONVAUTOENCODER, self).__init__()
        # encoder
        # conv output size : (insize - kernel_size +2*padding)/stride +1
        # maxpool output size :(insize - kernel_size +2*padding)/stride +1
        self.encode=nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),  #  32, 24, 24
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.3),
        #nn.CELU(True),
#        nn.Dropout2d(p=0.2),
        nn.MaxPool2d(poolsize, poolstride,poolpadding,return_indices=True)  # 32, 12, 12
        )
        # decoder
        # convtranspose size : (insize-1)*stride -2*padding + kernel_size + outpadding
        # max unpool size    : (insize-1)*stride -2*padding + kernel_size
        self.decode=nn.Sequential(
        nn.MaxUnpool2d(poolsize, poolstride, poolpadding),  # 32 ,24, 24
        nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride, padding),  # 1, 28, 28
        nn.BatchNorm2d(in_channels),
        nn.LeakyReLU(1)
        #nn.CELU(True)
        )

    def forward(self, x,encode=True,decode=True,noise=False,indices_and_deconvsize=None):


        if noise :
            noise=(torch.rand(x.size()))
            if torch.cuda.is_available():
                noise=noise.cuda()
            x = torch.add(x, 0.2 * (noise))

        
        if  encode :
            
            for i in range(len(self.encode)-1):
                x=self.encode[i](x)       
            
            deconvsizeout=x.size()
            x,indices=self.encode[len(self.encode)-1](x)
        
        
        
        if  decode :
            
            if indices_and_deconvsize !=None:
                    indices=indices_and_deconvsize[0]
                    deconvsizeout=indices_and_deconvsize[1]
            
            
            x=self.decode[0](x,indices,output_size=deconvsizeout)
            for i in range(1,len(self.decode)-1):
                x=self.decode[i](x)
        
    
    
        
        return x,indices,deconvsizeout
    
        
    
# %% 
# THIS CLASS DEFINES A MULTILAYER ARCHITECTURE

# THE CLASS ALLOWS TO GO UP AND DOWN THE ARCHITECTURE : encode-decode
        
# THE FORWARD DEFINITION IS COMPLEX, BUT ALLOWS ARTCHITECTURE FLEXIBILITY AND LAYER ACCES ON DEMAND
            
# ARGUEMENTS FOR THE FORWARD DEFINITION :        
# encode : default is true
# decode : default is true    
# encodelayer : which layer to encode
# decodelayer : which layer to decode
# linmap (linear mapping) : defautl is false
# noise : default is false        

            
class STACKEDAUTOENCODER(nn.Module):

    def __init__(self,outdim,chan,inputsize):
        super(STACKEDAUTOENCODER, self).__init__()
        
#       -parameters for conv = inchannels ,outchannels, kernelsize, stride, padding, poolsize, poolstride, poolpadding
#       paraeters for fully = insize, innersize




#  for michel 5 layers below

#        layer0   =   [chan,16,[1,9],[1,1],[0,0], [1,8],[1,8],[0,0]]
#        size0, nbfeatures0=conv_out_size(inputsize,*layer0[1:])
#        #
#        layer1   =   [16,16,[1,9],[1,1],[0,0],  [1,8],[1,8],[0,0]]
#        size1, nbfeatures1=conv_out_size(size0,*layer1[1:])
#
#        layer2   =   [16,16,[1,9],[1,1],[0,0],  [1,8],[1,8],[0,0]]
#        size2, nbfeatures2=conv_out_size(size1,*layer2[1:])
#        
#        layer3   =   [16,16,[1,9],[1,1],[0,0],  [1,8],[1,8],[0,0]]
#        size3, nbfeatures3=conv_out_size(size2,*layer3[1:])
#        

        
        ##        for 2d blobs
##        
        layer0   = [5,10]
        layer1   = [10,10]
        size0,finalnbfeatures  = layer0
        

        self.architecture=[
                layer0,
                layer1
#                layer2,
                ]
                                            
        self.layers=nn.ModuleList()
                
        for i in range(len(self.architecture)):
            
            if len(self.architecture[i])==8:
                self.layers.append(CONVAUTOENCODER(*self.architecture[i]))
            if len(self.architecture[i])==2:
                self.layers.append(AUTOENCODER(*self.architecture[i]))

        
        
        
        toplayer=AUTOENCODER(*[int(finalnbfeatures),outdim],top=True)
        
        self.layers.append(toplayer)
        
        self.architecture.append([int(finalnbfeatures),outdim])



    def forward(self, x,encode=True,decode=True,encodelayer=0,decodelayer=0,linmap=False,noise=False):

                      
        if encode :
            indices_deconvsize=[]
            inputs=[]
                                    
            for i in range(encodelayer+1):
                
                if i == decodelayer and noise ==True and decode ==True:
                    noise = True
                else :
                    noise = False



                # if current layer is fully connected, flatten input    
                
                if len(self.architecture[i])==2:
                    
                        x = x.view(x.size(0), -1)
                        inpu=x
                        
                        if linmap and i==encodelayer :
                            x =self.layers[i].encode[0](x)
                            inputs.append(inpu)
                        else:
                            x = self.layers[i](x,encode,False,noise)
                            indices_deconvsize.append(None)
                            inputs.append(inpu)
    
    
                # if current layer is convolutionnal but input is flat, reshape into square format
                
                elif len(x.size())==2:
                    
                        infeat=self.architecture[i][0]
                        x = x.view(x.size(0),infeat,*squaresize(x.size(1)/infeat))
                        inpu=x
                        
                        if linmap and i==encodelayer :
                            x =self.layers[i].encode[0](x)
                            inputs.append(inpu)
                        else:
                            x,ind,deconvsizeout = self.layers[i](x,encode,False,noise)
                            indices_deconvsize.append([ind,deconvsizeout])
                            inputs.append(inpu)
                            
                # if current layer is convolutionnal and input is not flat, do as normal     
                
                else:
                        inpu=x
                        
                        if linmap and i==encodelayer :
                            x =self.layers[i].encode[0](x)
                            inputs.append(inpu)
                        else:                        
                            x,ind,deconvsizeout = self.layers[i](x,encode,False,noise)
                            indices_deconvsize.append([ind,deconvsizeout])
                            inputs.append(inpu)
                            
            out=x
                    
        if decode :
            
            for i in reversed(range(decodelayer,encodelayer+1)):
            
                if len(self.architecture[i])==2:
                        x = x.view(x.size(0), -1)
                        x = self.layers[i](x,False,decode,noise)
    
                elif len(x.size())==2:
                        infeat=self.architecture[i][0]
                        x = x.view(x.size(0),infeat,*squaresize(x.size(1)/infeat) )
                        x,ind,deconvsizeout =self.layers[i](x,False,decode,noise,

                                                        indices_and_deconvsize=indices_deconvsize[i])
                else:
                        x,ind,deconvsizeout =self.layers[i](x,False,decode,noise,
                                                        indices_and_deconvsize=indices_deconvsize[i])
                
            out=x            
                        
        return inputs[decodelayer],out

    



# %% AUTOENCODER TRAINNING
                
#    LAYERS WIZE TRAINING, WITH OPTION TO GLOBALLY TRAIN ALL LAYERS
#    THE FUNCTION SAVES THE NET PARAMETERS AT EVERY EPOCH


def trainauto(fullname,VISUALIZE,NET_PATH,net,res50_conv,train_loader,test_loader,global_train=False):
        
    
    inputsize=train_loader.dataset.tensors[0].size()[2:4]    
    chan=net.layers[0].encode[0].weight.size()[1]
    outdim=len(net.layers[-1].encode[0].weight) 
    # to save all parameters at each epoch
    beforenetlayerlist=[]

    if VISUALIZE :        
        torch.save(beforenetlayerlist,NET_PATH + 'beforenetlayerlist' + fullname +'.pt')

    # LAYER PARAMETERS
    num_epochs = [10,10,30,30,30,30,30,30,30]
    patience=[3,3,3,3,3,3,3,3,3];
    lossimprocriteria=0.01
    learning_rate = [0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]
    weightdecay=[1e-5,1e-4,1e-4,1e-4,1e-4,1e-7,1e-7,1e-7,1e-7]

    
    #beforenet = type(net)(outdim,chan,inputsize) # get a new instance
    beforenet=net
    #beforenet.load_state_dict(net.state_dict())
    beforenet.train()
    
    
    if torch.cuda.is_available():
            beforenet.cuda()


#   PARAMETER WHEN GLOABAL TRAINING IS SET TO TRUE
    num_epochs_global=300  
    global_learningrate=0.01
    global_weightdecay=1e-12
    globalpatience=3;
    globallossimprocriteria=0.05
    
    layer_trainvalloss=[]
    layer_testvalloss=[]
    
    for layer in range(0,len(beforenet.architecture)-1):
        testvallos=[]
        trainvalloss=[]
        beforenetepochlist=[]
#        if VISUALIZE :        
#            torch.save(beforenetepochlist,NET_PATH + 'beforenetepochlist' + fullname +'.pt')


#       criterion = nn.CrossEntropyLoss() 
        criterion = nn.MSELoss()     
#        optimizer = torch.optim.SGD(beforenet.parameters(),lr=learning_rate[layer], weight_decay=weightdecay[layer],momentum=0.5)   # optimize all cnn parameters
#        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)#    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, beforenet.layers.parameters()),
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, beforenet.layers.parameters()),
        lr=learning_rate[layer],weight_decay=weightdecay[layer]) # optimize all cnn parameters#loss_func = nn.MSELoss()
        
    #    optimizer = torch.optim.Adam(beforenet.parameters(), lr=learning_rate,weight_decay=1e-4)   # optimize all cnn parameters
        
        for epoch in range(num_epochs[layer]):
        
        
        
            epochloss=[]    
            for step, (image, label) in enumerate(train_loader):
                                
                 if torch.cuda.is_available():
                     image=image.cuda()
                     label=label.cuda()

#                 image=res50_conv(image) 
#                 image=image.permute(0,3,2,1)
                     
                 beforenet.train()
                 inpu,outpu = beforenet(image,encode=True,decode=True,
                                   encodelayer=layer,decodelayer=layer,linmap=False,noise=True)           #  output
    
                 loss = criterion(outpu, inpu)   # 
                 optimizer.zero_grad()           # clear gradients for this training step
                 loss.backward()                 # backpropagation, compute gradients
                 optimizer.step()                # apply gradients
                 dataloss=loss.detach().data.cpu().numpy()
                 epochloss.append(dataloss)
                 del image, label,inpu, outpu,loss
                 
                             # save net state for this epoch
                 if VISUALIZE :
#                    beforenetepochlist=torch.load(NET_PATH + 'beforenetepochlist' + fullname +'.pt')
                    beforenet.eval()
                    outdim=len(beforenet.layers[-1].encode[0].weight)
                    netstate = type(beforenet)(outdim,chan,inputsize) # get a new instance
                    netstate.load_state_dict(beforenet.state_dict())
                    beforenetepochlist.append(netstate)
#                    torch.save(beforenetepochlist,NET_PATH + 'beforenetepochlist' + fullname +'.pt')
                    del netstate, outdim
                 
                  
            mean_epochloss=np.mean(epochloss)
            trainvalloss.append(mean_epochloss)
            
            if (epoch % patience[layer]==0):
#            if (epoch==epoch):

                print('Layer : '+ str(layer) + '  Epoch: ', epoch, '| train loss: %.4f' % mean_epochloss)
            del mean_epochloss            

            # check loss on validation set

            epochloss=[]
            for (image, label) in test_loader:
                 chan=image.size()[1]
                 if torch.cuda.is_available():
                     image=image.cuda()
                     label=label.cuda()
                     
#                 image=res50_conv(image) 
#                 image=image.permute(0,3,2,1)
                        
                 x=image.detach()
                 y=label
                 beforenet.eval()
                 inpu,outpu = beforenet(image,encode=True,decode=True,
                 encodelayer=layer,decodelayer=layer,linmap=False,noise=True)           #  output
                 loss = criterion(outpu, inpu) 
                 dataloss=loss.detach().data.cpu().numpy()
                 epochloss.append(dataloss)
                 del image,label, x, y,loss,inpu,outpu
              
              
            mean_epochloss=np.mean(epochloss)  
            testvallos.append(mean_epochloss)
            
            
            if (epoch % patience[layer]==0):
#            if (epoch==epoch):

                print('Layer : '+ str(layer) + '  Epoch: ', epoch, '| validation set loss: %.4f' % mean_epochloss)
            del epochloss
             
#            # save net state for this epoch
#            if VISUALIZE :
#                outdim=len(beforenet.layers[-1].encode[0].weight)
#                netstate = type(beforenet)(outdim,chan,inputsize) # get a new instance
#                netstate.load_state_dict(beforenet.state_dict())
#                beforenetepochlist.append(netstate)
#                del netstate, outdim
            
            
            
            gc.collect()
            
            if epoch >= patience[layer] and epoch % patience[layer]==0 :

#            if epoch >= patience[layer]:
                improv=abs(testvallos[epoch-patience[layer]]-testvallos[epoch])/testvallos[0]
                print(' relative % improvement to initial error since patience = '+ str(improv))
            
            
            
            # stop training if reaching plateau
            if epoch > patience[layer] and epoch % patience[layer]==0 and abs(testvallos[epoch-patience[layer]]-testvallos[epoch])/testvallos[0]<lossimprocriteria: 
                break
                    
        if VISUALIZE :
#                    beforenetepochlist=torch.load(NET_PATH + 'beforenetepochlist' + fullname +'.pt')
                    beforenetlayerlist=torch.load(NET_PATH + 'beforenetlayerlist' + fullname +'.pt')
                    beforenetlayerlist.append(beforenetepochlist)
                    torch.save(beforenetlayerlist,NET_PATH + 'beforenetlayerlist' + fullname+ '.pt' )
                    del beforenetepochlist, beforenetlayerlist
                    
        
        for param in beforenet.layers[layer].parameters():   
             param.requires_grad = False
             
        layer_testvalloss.append(testvallos)
        layer_trainvalloss.append(trainvalloss)
        plt.plot(testvallos)
#        plt.show() 
    
    np.save(NET_PATH + 'auto_trainloss' + fullname, layer_trainvalloss)
    np.save(NET_PATH + 'auto_testloss' + fullname, layer_testvalloss)
        
#    if VISUALIZE :
#            torch.save(beforenetlayerlist,NET_PATH + 'beforenetlayerlist' + fullname+ '.pt' )    

    
    
    
    # further training if global training is required : default is false
    
    if global_train:

        testvallos=[]
        trainvalloss=[]
             
        for i in range(len(list(beforenet.layers))-1):
            for param in beforenet.layers[i].parameters():   
                 param.requires_grad = True
             
        criterion = nn.MSELoss()     
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, beforenet.layers.parameters()),
        lr=global_learningrate,weight_decay=global_weightdecay)
            
        
        epochloss=[]
        for epoch in range(num_epochs_global):
            
            for step, (image, label) in enumerate(train_loader):

                
                 if torch.cuda.is_available():
                        image=image.cuda()
                        label=label.cuda()
                 beforenet.train()
                 inpu,outpu = beforenet(image,encode=True,decode=True,
                                   encodelayer=len(list(beforenet.layers))-2, decodelayer=0,linmap=False,
                                   noise=True)           #  output

                 loss = criterion(outpu, inpu)   # 
                 optimizer.zero_grad()           # clear gradients for this training step
                 loss.backward()                 # backpropagation, compute gradients
                 optimizer.step()                # apply gradients
                 dataloss=loss.detach().data.cpu().numpy()
                 epochloss.append(dataloss)
                 del image, label,inpu, outpu,loss
                 
    #        scheduler.step()   
            mean_epochloss=np.mean(epochloss)
            trainvalloss.append(mean_epochloss)
            
            if (epoch % globalpatience==0):
#            if (epoch==epoch):

                print('Epoch: ', epoch, '| train loss: %.4f' % mean_epochloss)
            
            
            # check loss on validation set

            epochloss=[]
            for (image, label) in test_loader:
                 if torch.cuda.is_available():
                     image=image.cuda()
                     label=label.cuda()
                 beforenet.eval()
                 inpu,outpu = beforenet(image,encode=True,decode=True,
                                  encodelayer=len(list(beforenet.layers))-2,decodelayer=0,linmap=False,
                                  noise=True)           #  output
                 loss = criterion(outpu, inpu)
                 dataloss=loss.detach().data.cpu().numpy()
                 epochloss.append(dataloss)
                 del image, label,inpu, outpu,loss
            
            
            mean_epochloss=np.mean(epochloss)
            testvallos.append(mean_epochloss)
            
            
            
            #         if (epoch%patience==0):
            if (epoch % globalpatience==0):
#            if (epoch==epoch):
                print('Epoch: ', epoch, '| validation set loss: %.4f' % mean_epochloss)
            del epochloss
            
            gc.collect()
            
            
            if epoch >= globalpatience and epoch % globalpatience==0:
                improv=abs(testvallos[epoch-globalpatience]-testvallos[epoch])/testvallos[0]
                print(' relative % improvement to initial error since patience = '+ str(improv))
            
            
            
            # stop training if reaching plateau
            if epoch > globalpatience and abs(testvallos[epoch-globalpatience]-testvallos[epoch])/testvallos[0]<globallossimprocriteria: 
                break
                        
        plt.plot(testvallos)
#        plt.show() 
        
        np.save(NET_PATH + 'auto_global_trainloss' + fullname, trainvalloss)
        np.save(NET_PATH + 'auto_global_testloss' + fullname, testvallos)
   

    return beforenet



# %%  TRAIN THE NETWORK FOR CATEGORIZATION

def trainclassifier(fullname,VISUALIZE,NET_PATH,classifier,res50_conv,inputsize,train_loader,test_loader):
    
#usecat,measurelayer,beforenet,
    
    chan=classifier.layers[0].encode[0].weight.size()[1]
    outdim=len(classifier.layers[-1].encode[0].weight)   
    
    classifier.train()
    # to save all parameters at each epoch
    classifierlist=[]
    
    if VISUALIZE :        
        torch.save(classifierlist,NET_PATH + 'classifierlist' + fullname +'.pt')
    
   
    if torch.cuda.is_available():
            classifier.cuda()
    
    
    # PARAMETERS
    onehotedvector=True  # scalar output vs vector output
    EPOCH = 10           # train the training data n times, to save time, we just train 1 epoch
    patience=3
    lossimprocriteria=0.01
    accuracycriteria=0.99
    testvallos=[]
    trainvalloss=[]
    accuracy_list=[]
    
    stimuliwizetrainloss=[]
    stimuliwizetestloss=[]
    stimuliwizetestaccuracy=[]
    
    LR = 0.01           # learning rate
    ey=torch.eye(outdim)
    
#   loss_func = nn.CrossEntropyLoss()                       # or CrossEntropyLoss if the target label is not one-hotted
    loss_func=nn.MSELoss() 
#   optimizer = torch.optim.SGD(classifier.parameters(), lr=LR,weight_decay=1e-4,momentum=0.2) # optimize all cnn parameters#loss_func = nn.MSELoss()
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)#    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, classifier.layers.parameters()),
#    
    optimizer = torch.optim.Adam(classifier.parameters(), lr=LR,
                                 weight_decay=1e-6)

    all_batch_accuracies=[]
    all_batch_loss=[]
    all_avgdist_cp=np.zeros((1,4))
    
    
    for epoch in range(EPOCH):
#         for itera, (image, label) in enumerate(train_loader):


         epochloss=[]
         for image, label in train_loader:   # gives batch data, normalize x when iterate train_loader


                y=label
                if onehotedvector:
                    label=ey[label]
                         

                if torch.cuda.is_available():
                    image=image.cuda()
                    label=label.cuda()
                    
                
#                image=res50_conv(image)
#                image=image.permute(0,3,2,1)
                
                classifier.train()
                inpu,outpu = classifier(image,encode=True,decode=False,
                              encodelayer=len(list(classifier.layers))-1,decodelayer=0,linmap=False) 
                
                loss = loss_func(outpu, label)   # cross entropy loss
                optimizer.zero_grad()           # clear gradients for this training step
                loss.backward()                 # backpropagation, compute gradients
                optimizer.step()                # apply gradients
                dataloss=loss.detach().data.cpu().numpy()
                epochloss.append(dataloss)
                del image, label,inpu, outpu,y,loss
                
                if VISUALIZE :
#                     classifierlist=torch.load(NET_PATH + 'classifierlist' + fullname +'.pt')
                     classifier.eval()
                     netstate = type(classifier)(outdim,chan,inputsize) # get a new instance
                     netstate.load_state_dict(classifier.state_dict())
                     classifierlist.append(netstate)
#                     torch.save(classifierlist,NET_PATH + 'classifierlist' + fullname +'.pt')
                     del netstate
    
         stimuliwizetrainloss.append(epochloss)
         mean_epochloss=np.mean(epochloss) 
         trainvalloss.append(mean_epochloss)
         
         if (epoch % patience==0):
#         if (epoch==epoch):
             print('Epoch: ', epoch, '| train loss: %.4f' % mean_epochloss)
         del mean_epochloss   
        
        # check loss on validation set
        
         
         epochloss=[]
         epochaccuracy=[]
         for (image, label) in test_loader:
             
             
             y=label                    
             if onehotedvector:
                label=ey[label]
             
             
             if torch.cuda.is_available():
                    image=image.cuda()
                    label=label.cuda()
                    
#             image=res50_conv(image) 
#             image=image.permute(0,3,2,1)
             
             x=image.detach()
             classifier.eval()
             inpu,outpu = classifier(x,encode=True,decode=False,
                              encodelayer=len(list(classifier.layers))-1,decodelayer=0,linmap=False)          

             loss = loss_func(outpu, label) 
             outpu=outpu.cpu()
             x=x.cpu()
             y=y.cpu()
             dataloss=loss.detach().data.cpu().numpy()
             epochloss.append(dataloss) 
             pred_y = torch.max(outpu, 1)[1] #.data.numpy()
#             accuracy = float((pred_y == y.data.numpy()).astype(int).sum()) / float(y.size(0))
#             accuracy = np.mean((pred_y == y).double().data.numpy())
             accuracy = (pred_y == y).double().data.numpy()
             epochaccuracy.append(accuracy)

                 
#             print(accuracy)
#             print(epochaccuracy)
             
#             all_batch_accuracies.append(accuracy)
#             all_batch_loss.append(dataloss)
#             autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin=\
#                        computecp(usecat,beforenet,classifier,test_loader,measurelayer=measurelayer)
#             dist=np.asarray([[autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin]])           
#             all_avgdist_cp=np.concatenate((all_avgdist_cp,dist),axis=0)
             
             del image, label,inpu, outpu,x,y,loss

         stimuliwizetestaccuracy.append(list(np.asarray(epochaccuracy).flatten()))
         mean_epochaccuracy=np.mean(epochaccuracy)
         accuracy_list.append(mean_epochaccuracy)
         if (epoch%patience==0):
#         if (epoch==epoch):
             print('Epoch: ', epoch, ' test accuracy: %.4f' % mean_epochaccuracy)
                 
                
         stimuliwizetestloss.append(epochloss)
         mean_epochloss=np.mean(epochloss)       
         testvallos.append(mean_epochloss) 
         
         if (epoch % patience==0):
#         if (epoch==epoch):

             print('Epoch: ', epoch, '| validation set loss: %.4f' % mean_epochloss) #mean_epochloss
             
         del epochloss
        

#         if VISUALIZE :
#             netstate = type(classifier)(outdim,chan,inputsize) # get a new instance
#             netstate.load_state_dict(classifier.state_dict())
#             classifierlist.append(netstate)
#             del netstate
     
         
         
         
         
         gc.collect()
         # stop training if reaching plateau
         
         if epoch >= patience and epoch % patience==0:
             improv=abs(testvallos[epoch-patience]-testvallos[epoch])/testvallos[0]
             print(' relative % improvement to initial error since patience = '+ str(improv))



         if epoch > patience and epoch % patience==0 and mean_epochaccuracy>accuracycriteria and abs(testvallos[epoch-patience]-testvallos[epoch])/testvallos[0]<lossimprocriteria: 
#         if mean_epochaccuracy>accuracycriteria:

             #             improv=(testvallos[epoch-patience]-testvallos[epoch])/testvallos[0]
#             print(' relative % improvement to initial error since patience = '+ str(improv))
             break

             
            
    if VISUALIZE :        
        torch.save(classifierlist,NET_PATH + 'classifierlist' + fullname +'.pt')
        
         
    plt.plot(testvallos)
#    plt.show()
    np.save(NET_PATH + 'classi_trainloss' + fullname, trainvalloss)
    np.save(NET_PATH + 'classi_testloss' + fullname, testvallos)
    np.save(NET_PATH + 'classi_accuracy' + fullname, accuracy_list)
    np.save(NET_PATH + 'stimuliwizetrainloss' + fullname, stimuliwizetrainloss)
    np.save(NET_PATH + 'stimuliwizetestloss' + fullname, stimuliwizetestloss)
    np.save(NET_PATH + 'stimuliwizetestaccuracy' + fullname, stimuliwizetestaccuracy)



#    np.save(NET_PATH + 'all_batch_accuracies' + fullname, all_batch_accuracies)
#    np.save(NET_PATH + 'all_batch_loss' + fullname, all_batch_loss)
#
#    np.save(NET_PATH + 'all_avgdist_cp' + fullname, all_avgdist_cp)

    
    return classifier


# %% COMPUTE CP
    
def computecp(usecat,beforenet,classifier,res50_conv,loader,measurelayer):
    
    
    def row_pairwise_distances(x, y):  # y=None is the original function
        
        
#        if len(x.size())==1:   # if batchsize =1 is one view does not work
#            x=torch.unsqueeze(x,dim=0)
#        
#        x_norm = (x**2).sum(1).view(-1, 1)
#        if y is not None:
#            if len(y.size())==1:   # if batchsize =1 is one view does not work
#                y=torch.unsqueeze(y,dim=0)
#            y_norm = (y**2).sum(1).view(1, -1)
#        else:
#            y = x
#            y_norm = x_norm.view(1, -1)
#
#        dist = torch.pow(x_norm + y_norm - 2 * torch.mm(x, torch.transpose(y, 0, 1)),1/2)
        
        
        # distance_matrix function needs non zero firt dimension (could happen becoause of unique below)
#        
        if len(x.size())==1:
            x=torch.unsqueeze(x,dim=0)

        if len(y.size())==1:
            y=torch.unsqueeze(y,dim=0)  

        dist=torch.tensor(distance_matrix(x,y))
        
        
        dist[dist != dist] = 0
        
        return dist
     
    
    
    if torch.cuda.is_available():
        classifier.cuda()
        beforenet.cuda()
    
    

#    beforenet.cpu()
#    classifier.cpu()
    
    beforenet.eval()
    classifier.eval()
    
    
    
    batch_avg=torch.tensor([])
    

    
    for (image, label) in loader:
        
            if torch.cuda.is_available():
                     image=image.cuda()
                     label=label.cuda()
        
#            image=res50_conv(image) 
#            image=image.permute(0,3,2,1)
            
            image=image.cpu()
            label=label.cpu()
            
        
       
            test_x=image                
            test_y=label
            

            
            # make sample are unique to eliminate zero distance bias for within
            test_x, indices =np.unique(test_x.numpy(),return_index=True,axis=0)
            test_x=torch.tensor(test_x)
            test_y=test_y[indices]
            
###            test_x[test_x != test_x] = 0  # remove nan

            
    


            x = test_x.clone().detach()            
            
            if torch.cuda.is_available():
                x=x.cuda()
            
            


            inpu,x = beforenet(x,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True)

            w=beforenet.layers[measurelayer].encode[0].weight.data.cpu()
            b=beforenet.layers[measurelayer].encode[0].bias.data.cpu()

            inpu=inpu.cpu()
            x=x.cpu()
            

            if len(beforenet.architecture[measurelayer])==2:
                autoactivation=(x-b)/torch.norm(w,dim=None)
            else:
                autoactivation=(x.sub(b[None,:,None,None]))/torch.norm(w,dim=None)
#            autoactivation=x
            autoactivation=autoactivation.view(autoactivation.size(0),-1).detach()
            
#            autoactivation[autoactivation != autoactivation] = 0
#            print(autoactivation)

            x = test_x.clone().detach()
            
            if torch.cuda.is_available():
                x=x.cuda()

            inpu,x = classifier(x,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True) 


            w=classifier.layers[measurelayer].encode[0].weight.data.cpu()
            b=classifier.layers[measurelayer].encode[0].bias.data.cpu()    
            inpu=inpu.cpu()
            x=x.cpu()


            if len(classifier.architecture[measurelayer])==2:
                classifieractivation=(x-b)/torch.norm(w,dim=None)
            else:
                classifieractivation=(x.sub(b[None,:,None,None]))/torch.norm(w,dim=None) 
#            classifieractivation=x
            classifieractivation=classifieractivation.view(classifieractivation.size(0),-1).detach()
#            print(torch.nonzero(torch.isnan(classifieractivation)))
            
#            classifieractivation[classifieractivation != classifieractivation] = 0
#            print(classifieractivation)

            autoactivationbycat=[None] * len(usecat)
            classifieractivationbycat=[None] * len(usecat)
            autoavgdistwithinbycat=torch.empty(len(usecat))
            classifieravgdistwithinbycat=torch.empty(len(usecat))

            autoavgdistbetweenAll=torch.empty(int((len(usecat)*(len(usecat)-1))/2))
            classifieravgdistbetweenAll=torch.empty(int((len(usecat)*(len(usecat)-1))/2))


            for i in range(len(usecat)):
                index=torch.squeeze(torch.nonzero(test_y.data==usecat[i]))
                autoactivationbycat[i]=autoactivation[index]
                classifieractivationbycat[i]=classifieractivation[index]



            # Within
            for i in range(len(usecat)):
                autodistmatwithin=row_pairwise_distances(autoactivationbycat[i],autoactivationbycat[i])
#                print(autoactivationbycat[i].size())
#                autodistmatwithin=torch.tensor(distance_matrix(autoactivationbycat[i],autoactivationbycat[i]))
                triumat=torch.triu(autodistmatwithin,diagonal=1)
#                autoavgdistwithin=torch.sum(triumat)/torch.nonzero(triumat.size(0))
                autoavgdistwithin=torch.sum(triumat)/torch.nonzero(triumat).size(0)
#                autoavgdistwithin=torch.sum(triumat)/(triumat.size(0)*(triumat.size(0)-1)/2)
                

#                autoavgdistwithin=torch.mean(triumat[torch.nonzero(triumat)])
#                autoavgdistwithin[autoavgdistwithin != autoavgdistwithin] = 0
#                print(autoavgdistwithin)
                autoavgdistwithinbycat[i]=autoavgdistwithin

                classifierdistmatwithin=row_pairwise_distances(classifieractivationbycat[i],classifieractivationbycat[i])
#                classifierdistmatwithin=torch.tensor(distance_matrix(classifieractivationbycat[i],classifieractivationbycat[i]))
                triumat=torch.triu(classifierdistmatwithin,diagonal=1)
                classifieravgdistwithin=torch.sum(triumat)/torch.nonzero(triumat).size(0)
#                classifieravgdistwithin=torch.sum(triumat)/(triumat.size(0)*(triumat.size(0)-1)/2)
                
#                classifieravgdistwithin=torch.sum(triumat)
#                classifieravgdistwithin[classifieravgdistwithin != classifieravgdistwithin] = 0
#                print(classifieravgdistwithin)
                classifieravgdistwithinbycat[i]=classifieravgdistwithin


            autoavgdistwithin=torch.mean(autoavgdistwithinbycat)
            classifieravgdistwithin=torch.mean(classifieravgdistwithinbycat)




            # Between
            k=0
            for i in range(len(usecat)):
                for j in range(i):
                    autodistmatbetween=row_pairwise_distances(autoactivationbycat[i],autoactivationbycat[j])
#                    autodistmatbetween=torch.tensor(distance_matrix(autoactivationbycat[i],autoactivationbycat[j]))
                    if len(test_x)==4:
                        autodistmatbetween=torch.diag(autodistmatbetween)
                        autoavgdistbetween=torch.sum(autodistmatbetween)/(len(autodistmatbetween))
#                        autoavgdistbetween=torch.mean(autodistmatbetween)
                    else:
                        autoavgdistbetween=torch.sum(autodistmatbetween)/(autodistmatbetween.size(0)**2)

#                        autoavgdistbetween=torch.mean(autodistmatbetween)                        
                    autoavgdistbetweenAll[k]=autoavgdistbetween

                    classifierdistmatbetween=row_pairwise_distances(classifieractivationbycat[i],classifieractivationbycat[j])
#                    classifierdistmatbetween=torch.tensor(distance_matrix(classifieractivationbycat[i],classifieractivationbycat[j]))
                    if len(test_x)==4:
                        classifierdistmatbetween=torch.diag(classifierdistmatbetween)
                        classifieravgdistbetween=torch.sum(classifierdistmatbetween)/(len(classifierdistmatbetween))
#                        classifieravgdistbetween=torch.mean(classifierdistmatbetween)

                    else:
                        classifieravgdistbetween=torch.sum(classifierdistmatbetween)/(classifierdistmatbetween.size(0)**2)
#                        classifieravgdistbetween=torch.mean(classifierdistmatbetween)
                    classifieravgdistbetweenAll[k]=classifieravgdistbetween


                    k=k+1

            autoavgdistbetween=torch.mean(autoavgdistbetweenAll)
            classifieravgdistbetween=torch.mean(classifieravgdistbetweenAll)




            avg=torch.tensor([autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])
            batch_avg=torch.cat((batch_avg,torch.unsqueeze(avg,dim=0)),dim=0)
            
            del autodistmatwithin,triumat,autodistmatbetween
            gc.collect()
        
            
    
    
    
    autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin = [*torch.mean(batch_avg,dim=0)]
    
    
    

    return autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin





# %%
def visualize_reconstruction(network,fulltrain_loader):    
    
    
    encodeuptolayer=0
        
    for (sample, label) in fulltrain_loader:
        im=np.random.randint(len(sample))
                
    inpu=torch.unsqueeze(sample[im],0).detach()
    x=inpu.clone()
    noise=(torch.rand(x.size()))-0
    x = torch.add(x, 0.1* noise)
    
    network.eval()
    network.cpu()

    x,outpu = network(x,encode=True,decode=True,encodelayer=encodeuptolayer,
                                   decodelayer=0,linmap=False)           #  output

    outpu=outpu.cpu()

    
    plt.ion()
    
    img=torch.cat((inpu,x,outpu),dim=0)
    #img=torch.unsqueeze(img,1)
    img=img.detach()
    img = make_grid(img)
    plt.imshow(img.permute(1, 2, 0).numpy())
    plt.show()
    
    #plt.imsave('./pytorch_codes/images/denoise.png',img.permute(1, 2, 0))
# %%

def visualize_firstlayer_filters(network):    
    
    kernels = network.layers[0].encode[0].weight.detach()
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    img = make_grid(kernels)
    fig=plt.imshow(img.permute(1, 2, 0).cpu().numpy())
    plt.show()
    
    
# %% VISUALIZE LAYER EVOLUTION
import matplotlib    
import matplotlib.animation as animation
from matplotlib import cm
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA


    
def visualize_layer(loader,res50_conv, beforenetlist, classifierlist, layerid):
    



    fig=plt.figure()
#    frames=np.concatenate((np.arange(len(beforenetlist[layerid])),np.arange(len(classifierlist))))
#    frames=np.arange(0,len(classifierlist),100)
    framess=np.floor(np.linspace(0,len(classifierlist),len(classifierlist)-1)).astype(int)

    numb_dict = {1: 'A', 0: 'B'}

        
    def anifunction(i):
        
                
        
        for (image, label) in loader:     
            
            if torch.cuda.is_available():
                image=image.cuda()
                
        
            image=res50_conv(image) 
            image=image.permute(0,3,2,1)
            
#            image=image.cpu()
#            label=label.cpu()    
                
                
            lab=label                    
            X=image.clone()
        
        
        
#        if (frames[i]==i):
#            beforenet=beforenetlist[layerid][i]
#            if torch.cuda.is_available():
#                beforenet.cuda()
##            beforenet.cpu()
#            inpu,X = beforenet(X,encode=True,decode=False,
#                              encodelayer=layerid,decodelayer=0,linmap=False)     
#            X = X.view(X.size(0), -1) 
#            title='AUTOENCODER : training trial = ' + str(i)
#
#        else:
        classifier=classifierlist[framess[i]]

        if torch.cuda.is_available():
            classifier.cuda()
#            classifier.cpu()
        inpu,X = classifier(X,encode=True,decode=False,
                          encodelayer=layerid,decodelayer=0,linmap=False)
        
        w=classifier.layers[layerid].encode[0].weight.data #.cpu()
        b=classifier.layers[layerid].encode[0].bias.data #.cpu()  
        X=(X-b)/torch.norm(w,dim=None)
        
        X = X.view(X.size(0), -1) 
        title='CLASSIFIER : training trial = ' + str(framess[i+1]) 
        

        plt.clf()
        X=X.cpu()
#        reduce = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#        seed = np.random.RandomState(seed=1)
        reduce = MDS(n_components=2,dissimilarity="euclidean",random_state=10)
#        reduce = LocallyLinearEmbedding(n_components=2,random_state=10)
#        reduce = PCA(n_components=2,random_state=100)


        
        plot_only = 500
#        plot_only=np.sort(np.random.choice(X.size(0),300,replace=False))
        low_dim_embs = reduce.fit_transform(X.data.numpy().astype(np.float64)[:plot_only, :])
        lab = lab.numpy()[:plot_only]
        
        
        xcoor, ycoor = low_dim_embs[:, 0], low_dim_embs[:, 1]
#        xcoor = low_dim_embs[:, 0]
#        ycoor= np.zeros(np.shape(xcoor))  # if using  number of components = 1 

#       plt.scatter(nx, ny, c = cdict[j], label = j, s = 100)
        
        ax2  = plt.subplot(1,1,1)
        for x, y, s in zip(xcoor, ycoor, lab):
#            c = cm.rainbow(int(255 * s / 9));
            c = cm.rainbow(int(255*s/9)); 

            plt.text(x, y,numb_dict[s], backgroundcolor=c, fontsize=8,\
                          bbox=dict(facecolor=c, edgecolor=c, boxstyle='round'))
            
            plt.xlim(xcoor.min(), xcoor.max()); plt.ylim(ycoor.min(), ycoor.max());
#            plt.xlim(-100, 100); plt.ylim(-100, 100);
            plt.text(0.5, 1.08, title,horizontalalignment='center',fontsize=16,transform = ax2.transAxes)
#            plt.title(title)
        
#        plt.imshow(np.matlib.repmat(-(xcoor-np.mean(xcoor)),100,1),cmap='rainbow')  
            
#        plt.scatter(xcoor, ycoor,zcoor, c=lab, cmap=matplotlib.colors.ListedColormap(cdict))    

    
    ani = animation.FuncAnimation(fig, anifunction,frames=len(framess), interval=500, blit=False,repeat_delay=1000)
    
    return ani
    
    


# %% VISUALIZE CP EVOLUTION
import matplotlib    
import matplotlib.animation as animation
from matplotlib import cm
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
#from net_classes_and_functions import computecp

    
def visualize_cp(usecat,loader, beforenetlist, classifierlist, layerid):
    

    fig=plt.figure()
    frames=np.arange(len(classifierlist))
        
    def anifunction(i):
        plt.clf() 
        autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin=computecp(usecat,beforenetlist[-1][-1],classifierlist[i],loader,measurelayer=layerid)
        plotcp(autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin)        

    ani = animation.FuncAnimation(fig, anifunction,frames=len(frames), interval=500, blit=False,repeat_delay=1000)
    
    return ani
    

# %% VISUALIZE WEIGHT ASSIGNED TO INPUT DIMENSIONS: 
# ONLY FOR FULLLY CONNECTED FIRST LAYER AND SMALL NET

import numpy as np

def visualize_input_dimensions(w):

    
#    w=network.layers[layer].encode[0].weight
    numneuron,dim=w.size()
    w=w.data.numpy()
    
    for i in range(numneuron):
      plt.subplot(2,np.round(numneuron/2),i+1)
      plt.bar(np.arange(dim),np.abs(w[i,:]))
      plt.title('neuron '+ str(i),fontSize= 10)
    plt.show()
    
    
    bar=plt.bar(np.arange(dim),np.sqrt(np.sum(w**2,0)))
    plt.title('Combined weights of all neurons',fontSize= 14)
    return bar
#    plt.show()
    
# %%
    
def checkclassoutput(classifier, test_loader, numberoftestsamples):
    
    
    classifier.eval()
    classifier.cpu()


    for (image, label) in test_loader:
             test_x=image
             test_y=label


    inpu,out = classifier(test_x[:20],encode=True,decode=False,
                              encodelayer=len(list(classifier.layers))-1,decodelayer=0,linmap=False)

    inpu=inpu.cpu()
    out=out.cpu()

    pred_y = torch.max(out, 1)[1].data.numpy()
    print(pred_y, 'Net predicted class')
    
    print(test_y[:numberoftestsamples].numpy(), 'Actual class')
    

# %%
def visualize_top_layers_learned_features(network,fulltrain_loader,encodelayer,filterid):

    
    network.cpu()
    network.eval()    
    if torch.cuda.is_available():
        network.cuda()
        
    for (sample, label) in fulltrain_loader:
          im=np.random.randint(len(sample))
          
    inpu=torch.unsqueeze(sample[im],0)
#    inpu = Variable(0.00001*torch.randn(inpu.size()), requires_grad=True)    
#    noise = Variable(torch.randn(inpu.size()), requires_grad=True)
#    inpu = inpu + 0*noise
        
    
    inpu=Variable(inpu,requires_grad=True)
    if torch.cuda.is_available():
        inpu=Variable(inpu.cuda(),requires_grad=True)
        
    num_epochs=1500
    weight_decay=1e-2
    lr=0.01

#    network.zero_grad()
#    optimizer=torch.optim.Adam([inpu], lr=lr,weight_decay=weight_decay)
#    criterion = nn.MSELoss()
#    target=torch.tensor([1,0],dtype=torch.float)
#    if torch.cuda.is_available():
#        target=target.cuda()
             
    for epoch in range(num_epochs):

                     
        
             inp,outpu = network(inpu,encode=True,decode=False,encodelayer=encodelayer,
                               decodelayer=0,linmap=False
                               )           #  output
             
#             inpu.requires_grad = True
             loss = outpu[0,filterid].norm()
             loss.backward()
             inpu.data = inpu.data + lr * inpu.grad.data
             
#             loss = criterion(outpu, target)   # 
#             optimizer.zero_grad()           # clear gradients for this training step
#             loss.backward()                 # backpropagation, compute gradients
#             optimizer.step()                # apply gradients

             if (epoch%100==0):
                  print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())
             
             
    inpu=inpu.cpu()       
    plt.ion()
    plt.imshow(torch.squeeze(inpu).detach().cpu().numpy(),cmap='gray')
    
    
    #plt.savefig('./pytorch_codes/savedfigures/input.png')
    
    plt.show()


# %%
    

def conv_out_size(size=None,outchannels=None,kernelsize=None,stride=None,padding=None,poolsize=None,
                  poolstride=None, poolpadding=None):   
    
    
    if kernelsize==None:
        sizE,nbfeatures = size,outchannels
    else:
    
        x= int((size[0] - kernelsize[0] +2*padding[0])/stride[0] +1)
        y= int((size[1] - kernelsize[1] +2*padding[1])/stride[1] +1)
        
        x= int((x - poolsize[0] +2*poolpadding[0])/poolstride[0] +1)
        y= int((y - poolsize[1] +2*poolpadding[1])/poolstride[1] +1)
        
        nbfeatures=x*y*outchannels
        
        sizE=[x,y]
    return sizE,nbfeatures


def squaresize(n):
    
    pairs=[[x, int(n/x)] for x in range(1, int(math.sqrt(n))+1) if n % x == 0]
    p=np.argmin(np.sum(pairs,1))
    size=pairs[p]

    return size


def plotcp(autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin):
    
    plt.ylabel("Average pairwise distance",fontsize=15)
    
    x = np.linspace(0, 2, 2)
    
    im=plt.plot(x, [autoavgdistwithin ,classifieravgdistwithin],'g',
     x, [autoavgdistbetween ,classifieravgdistbetween],'b',linewidth=3.0)
    plt.legend([im[0],im[1]],['within','between'],bbox_to_anchor=(0.6,0.6),fontsize=15)

    
    im=plt.ylim(np.min([autoavgdistbetween.data ,classifieravgdistbetween.data,
                       autoavgdistwithin.data ,classifieravgdistwithin.data])-1,
                np.max([autoavgdistbetween.data ,classifieravgdistbetween.data,
                       autoavgdistwithin.data ,classifieravgdistwithin.data])+1)
                        
            
    plt.xlim(-1, 3)
    plt.xticks([0, 2], ["Before", "After"])


    #compression=classifieravgdistwithin-autoavgdistwithin
    #separation= classifieravgdistbetween-compression
    #GLOBALCP = separation-(classifieravgdistwithin-autoavgdistwithin)
    #INITIAL_SEPARATION = autoavgdistbetween-autoavgdistwithin
    #print('gloabal cp = ' , GLOBALCP.numpy())
    #print('Initial separation = ' , INITIAL_SEPARATION.numpy())
    #print('separation = ' , separation.numpy())
    #print('compression = ' , compression.numpy())
    
    
    
# %% rainbow animation


import matplotlib    
import matplotlib.animation as animation
from matplotlib import cm
from sklearn.manifold import TSNE
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



    
def visualize_rainbow(classifierlist, layerid):
    


#    plt.figure()
    fig,ax=plt.subplots()
#    ax.set_xticks(np.linspace(0,5000,6))
#    ax.set_xticklabels(['0','0.2','0.4','0.6','0.8','1'])
#    fig = plt.figure()
#    ax1 = fig.add_subplot(1,2,1)
#    ax2 = fig.add_subplot(1,2,2)
#    im2, = ax2.plot([], [], color=(0,0,1))
#    ax=fig.add_subplot(111)
#    frames=np.concatenate((np.arange(len(beforenetlist[layerid])),np.arange(len(classifierlist))))
    frames=np.arange(0,len(classifierlist),1)
    


        
    def animatefunction(i):
        
        
               
        N=1
        r=0
        difference=0.05
        #k1=0
        #k2=1
        #k3=2
        ##measurelayer=2
        
        ma=1
        
        pts=5000
        
        for k in range(1,2):
            
            if k==1:
                pstart=1
            else:
                pstart=2
                
            
            
            for p in range(pstart,k+1):
                

                chan=1
                
                
                
                classifier=classifierlist[frames[i]]
                if torch.cuda.is_available():
                    classifier.cuda()
#                    classifier.cpu()

                classifier.eval()
        
        
        
                for measurelayer in range(layerid,layerid+1):
                    
                    points = np.random.uniform(0,ma,[pts,N])
                    for j in range(k):
                        points[:,j]=np.arange(0,ma,ma/pts)
        
                
                #    points[:,k]=np.random.uniform(0,1-difference,5000)
                    idx=np.argsort(points[:,0], axis=0)  # not necessary if already sorted "arange"
                    points=points[idx]
                    
                    equidispoint=np.copy(points)
                    for j in range(k):
                        equidispoint[:,j]=equidispoint[:,j]+difference
                
                    
                    points=torch.tensor(points,dtype=torch.float)  
                    points=torch.unsqueeze(points,dim=1)
                    points=torch.unsqueeze(points,dim=1)
                    
                    equidispoint=torch.tensor(equidispoint,dtype=torch.float)  
                    equidispoint=torch.unsqueeze(equidispoint,dim=1)
                    equidispoint=torch.unsqueeze(equidispoint,dim=1)
                    
                    
                    points=points.clone().detach()
                    equidispoint=equidispoint.clone().detach()
                    
                                        
                    
                    
                    inpu,out = classifier(points,encode=True,decode=False,
                                    encodelayer=measurelayer,decodelayer=0,linmap=True)
                    
                    
                    w=classifier.layers[measurelayer].encode[0].weight.data.cpu()
                    b=classifier.layers[measurelayer].encode[0].bias.data.cpu()
                    inpu=inpu.cpu()
                    out=out.cpu()        
                    pointsclassifier=(out-b)/torch.norm(w,dim=None)
                    
                    
                    
                    inpu,out = classifier(equidispoint,encode=True,decode=False,
                                    encodelayer=measurelayer,decodelayer=0,linmap=True)
                    w=classifier.layers[measurelayer].encode[0].weight.data.cpu()
                    b=classifier.layers[measurelayer].encode[0].bias.data.cpu()
                    inpu=inpu.cpu()
                    out=out.cpu()        
                    equidispointclassifier=(out-b)/torch.norm(w,dim=None)
                    
                    
                    diffclass=equidispointclassifier-pointsclassifier
                    
                    
                    
                    spec1=torch.norm(diffclass,dim=1).detach().numpy()
        #            spec1=torch.norm(equidispointclassifier,dim=1).detach().numpy()
        #            spec2=torch.norm(pointsclassifier,dim=1).detach().numpy()
        
        
                #    spec=torch.norm(diffclass,dim=1).detach().numpy()
                    
        #            spec=scipy.ndimage.filters.gaussian_filter(spec, 40, order=0,output=None, mode='reflect', cval=0.0, truncate=12.0)
                    
                    #plt.plot(spec)
#                    fig, ax = plt.subplots()
                    plt.clf()
#                    im1.plot(spec1-np.mean(spec1))
#                    im1=ax1.plot(spec1-np.mean(spec1))
#                    im2, = ax2.plot([], [], color=(0,0,1))
                    
#                    plt.imshow(np.matlib.repmat(-(spec1-np.mean(spec1)),1000,1))
                    plt.plot(spec1-np.mean(spec1))

        #            plt.plot(spec2-np.mean(spec2))
#                    im2.set_xdata(np.linspace(0,1,5))
        
                    
#                    ax.xaxis.set_major_locator(plt.FixedLocator(np.linspace(0,5000,5)))
#                    ax.xaxis.set_major_formatter(plt.FixedFormatter(np.linspace(0,1,5)))
#                    plt.savefig(FIGURE_PATH  +'2dcontinum_layer' +str(measurelayer)+ name +'.png',bbox_inches = 'tight',
#                    pad_inches = 0)
        return 
                    
    ani = animation.FuncAnimation(fig, animatefunction,frames=len(frames), interval=500, blit=False,repeat_delay=1000)
    
    return ani
    


# %% COMPUTE CP
    
def computecp_raw(usecat,beforenet,classifier,res50_conv,loader,measurelayer):
    import torch.nn.functional as f
    
    
    def row_pairwise_distances(x, y):  # y=None is the original function
        
        
#        if len(x.size())==1:   # if batchsize =1 is one view does not work
#            x=torch.unsqueeze(x,dim=0)
#        
#        x_norm = (x**2).sum(1).view(-1, 1)
#        if y is not None:
#            if len(y.size())==1:   # if batchsize =1 is one view does not work
#                y=torch.unsqueeze(y,dim=0)
#            y_norm = (y**2).sum(1).view(1, -1)
#        else:
#            y = x
#            y_norm = x_norm.view(1, -1)
#
#        dist = torch.pow(x_norm + y_norm - 2 * torch.mm(x, torch.transpose(y, 0, 1)),1/2)
        
        
        # distance_matrix function need non zero firt dimension (could happen becoause of unique below)
#        
        if len(x.size())==1:
            x=torch.unsqueeze(x,dim=0)

        if len(y.size())==1:
            y=torch.unsqueeze(y,dim=0)  

        dist=torch.tensor(distance_matrix(x,y))
        
        
        dist[dist != dist] = 0
        
        return dist
     
    
    
    if torch.cuda.is_available():
        classifier.cuda()
        beforenet.cuda()
    
    

#    beforenet.cpu()
#    classifier.cpu()
    
    beforenet.eval()
    classifier.eval()
    
    
    
    batch_avg=torch.tensor([])
    

    
    for (image, label) in loader:
        
            rawactivation=image.view(image.size(0),-1).detach()
            rownorm=torch.norm(rawactivation,dim=1)
#            f.normalize(rawactivation, p=2, dim=0)
            rawactivation=rawactivation/torch.max(rownorm)
        
        
            if torch.cuda.is_available():
                     image=image.cuda()
                     label=label.cuda()
        
#            image=res50_conv(image) 
#            image=image.permute(0,3,2,1)
            
            image=image.cpu()
            label=label.cpu()
            

            
        
       
            test_x=image                
            test_y=label
            


            

            
            # make sample are unique to eliminate zero distance bias for within
            test_x, indices =np.unique(test_x.numpy(),return_index=True,axis=0)
            test_x=torch.tensor(test_x)
            test_y=test_y[indices]
            
###            test_x[test_x != test_x] = 0  # remove nan

            
            print(test_x.size())


            x = test_x.clone().detach()            
            
            if torch.cuda.is_available():
                x=x.cuda()
            
            


            inpu,x = beforenet(x,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True)

            w=beforenet.layers[measurelayer].encode[0].weight.data.cpu()
            b=beforenet.layers[measurelayer].encode[0].bias.data.cpu()

            inpu=inpu.cpu()
            x=x.cpu()
            

            if len(beforenet.architecture[measurelayer])==2:
                autoactivation=(x-b)/torch.norm(w,dim=None)
            else:
                autoactivation=(x.sub(b[None,:,None,None]))/torch.norm(w,dim=None)
#            autoactivation=x
            autoactivation=autoactivation.view(autoactivation.size(0),-1).detach()
            
            rownorm=torch.norm(autoactivation,dim=1)
            autoactivation=autoactivation/torch.max(rownorm)
#            f.normalize(autoactivation, p=2, dim=0)

#            print(torch.max(rownorm))
            
            
            
#            autoactivation[autoactivation != autoactivation] = 0
#            print(autoactivation)

            x = test_x.clone().detach()
            
            if torch.cuda.is_available():
                x=x.cuda()

            inpu,x = classifier(x,encode=True,decode=False,
                            encodelayer=measurelayer,decodelayer=0,linmap=True) 


            w=classifier.layers[measurelayer].encode[0].weight.data.cpu()
            b=classifier.layers[measurelayer].encode[0].bias.data.cpu()    
            inpu=inpu.cpu()
            x=x.cpu()


            if len(classifier.architecture[measurelayer])==2:
                classifieractivation=(x-b)/torch.norm(w,dim=None)
            else:
                classifieractivation=(x.sub(b[None,:,None,None]))/torch.norm(w,dim=None) 
#            classifieractivation=x
            classifieractivation=classifieractivation.view(classifieractivation.size(0),-1).detach()
#            print(torch.nonzero(torch.isnan(classifieractivation)))
            
#            rownorm=torch.norm(classifieractivation,dim=1)
            classifieractivation=classifieractivation/torch.max(rownorm)
#            f.normalize(classifieractivation, p=2, dim=0)

            
#            print(torch.max(rownorm))



            autoactivationbycat=[None] * len(usecat)
            classifieractivationbycat=[None] * len(usecat)
            autoavgdistwithinbycat=torch.empty(len(usecat))
            classifieravgdistwithinbycat=torch.empty(len(usecat))

            autoavgdistbetweenAll=torch.empty(int((len(usecat)*(len(usecat)-1))/2))
            classifieravgdistbetweenAll=torch.empty(int((len(usecat)*(len(usecat)-1))/2))
            
            
            
            rawactivationbycat=[None] * len(usecat)
            rawavgdistwithinbycat=torch.empty(len(usecat))

            rawavgdistbetweenAll=torch.empty(int((len(usecat)*(len(usecat)-1))/2))


            for i in range(len(usecat)):
                index=torch.squeeze(torch.nonzero(test_y.data==usecat[i]))
                
                rawactivationbycat[i]=rawactivation[index]
                autoactivationbycat[i]=autoactivation[index]
                classifieractivationbycat[i]=classifieractivation[index]




            # Within
            for i in range(len(usecat)):
                
                rawdistmatwithin=row_pairwise_distances(rawactivationbycat[i],rawactivationbycat[i])
                triumat=torch.triu(rawdistmatwithin,diagonal=1)
                rawavgdistwithin=torch.sum(triumat)/torch.nonzero(triumat).size(0)
                rawavgdistwithinbycat[i]=rawavgdistwithin

                
                
                autodistmatwithin=row_pairwise_distances(autoactivationbycat[i],autoactivationbycat[i])
                triumat=torch.triu(autodistmatwithin,diagonal=1)
                autoavgdistwithin=torch.sum(triumat)/torch.nonzero(triumat).size(0)
                autoavgdistwithinbycat[i]=autoavgdistwithin

                classifierdistmatwithin=row_pairwise_distances(classifieractivationbycat[i],classifieractivationbycat[i])
                triumat=torch.triu(classifierdistmatwithin,diagonal=1)
                classifieravgdistwithin=torch.sum(triumat)/torch.nonzero(triumat).size(0)
                classifieravgdistwithinbycat[i]=classifieravgdistwithin

            rawavgdistwithin=torch.mean(rawavgdistwithinbycat)
            autoavgdistwithin=torch.mean(autoavgdistwithinbycat)
            classifieravgdistwithin=torch.mean(classifieravgdistwithinbycat)




            # Between
            k=0
            for i in range(len(usecat)):
                for j in range(i):
                    
                    
                    rawdistmatbetween=row_pairwise_distances(rawactivationbycat[i],rawactivationbycat[j])
                    if len(test_x)==4:
                        rawdistmatbetween=torch.diag(rawdistmatbetween)
                        rawavgdistbetween=torch.sum(rawdistmatbetween)/(len(rawdistmatbetween))
                    else:
                        rawavgdistbetween=torch.sum(rawdistmatbetween)/(rawdistmatbetween.size(0)**2)

                    rawavgdistbetweenAll[k]=rawavgdistbetween




                    autodistmatbetween=row_pairwise_distances(autoactivationbycat[i],autoactivationbycat[j])
                    if len(test_x)==4:
                        autodistmatbetween=torch.diag(autodistmatbetween)
                        autoavgdistbetween=torch.sum(autodistmatbetween)/(len(autodistmatbetween))
                    else:
                        autoavgdistbetween=torch.sum(autodistmatbetween)/(autodistmatbetween.size(0)**2)

                    autoavgdistbetweenAll[k]=autoavgdistbetween



                    classifierdistmatbetween=row_pairwise_distances(classifieractivationbycat[i],classifieractivationbycat[j])
                    if len(test_x)==4:
                        classifierdistmatbetween=torch.diag(classifierdistmatbetween)
                        classifieravgdistbetween=torch.sum(classifierdistmatbetween)/(len(classifierdistmatbetween))

                    else:
                        classifieravgdistbetween=torch.sum(classifierdistmatbetween)/(classifierdistmatbetween.size(0)**2)
                    classifieravgdistbetweenAll[k]=classifieravgdistbetween


                    k=k+1

            rawavgdistbetween=torch.mean(rawavgdistbetweenAll)
            autoavgdistbetween=torch.mean(autoavgdistbetweenAll)
            classifieravgdistbetween=torch.mean(classifieravgdistbetweenAll)




            avg=torch.tensor([rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin])
            batch_avg=torch.cat((batch_avg,torch.unsqueeze(avg,dim=0)),dim=0)
            
            del autodistmatwithin,triumat,autodistmatbetween,rawdistmatwithin,rawdistmatbetween
            gc.collect()
        
            
    
    
    
    rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin = [*torch.mean(batch_avg,dim=0)]
    
    
    

    return rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin




def plotcp_raw(rawavgdistbetween,rawavgdistwithin,autoavgdistbetween,autoavgdistwithin,classifieravgdistbetween,classifieravgdistwithin):
    
    plt.ylabel("Average pairwise distance",fontsize=15)
    
    x = np.linspace(0, 3, 3)#[0:3]
    
    im=plt.plot(x, [rawavgdistwithin,autoavgdistwithin ,classifieravgdistwithin],'g',
     x, [rawavgdistbetween,autoavgdistbetween ,classifieravgdistbetween],'b',linewidth=3.0)
    plt.legend([im[0],im[1]],['within','between'],bbox_to_anchor=(0.6,0.6),fontsize=15)

    
    im=plt.ylim(np.min([rawavgdistbetween.data,autoavgdistbetween.data ,classifieravgdistbetween.data,
                       rawavgdistwithin.data,autoavgdistwithin.data ,classifieravgdistwithin.data])-1,
                np.max([rawavgdistbetween.data,autoavgdistbetween.data ,classifieravgdistbetween.data,
                       rawavgdistwithin.data,autoavgdistwithin.data ,classifieravgdistwithin.data])+1)
                        
            
    plt.xlim(-1, 4)
    plt.xticks([0,1.5 ,3], ["Raw","Before", "After"])
    
