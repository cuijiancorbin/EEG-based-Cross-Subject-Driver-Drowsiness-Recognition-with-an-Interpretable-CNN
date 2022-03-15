# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN

This file implement the visualization technique proposed in the paper.
The extracted dataset is available from:
    https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687
if you have any problem on using the codes, pls contact Dr. Cui Jian at cuij0006@ntu.edu.sg
"""
import torch
import scipy.io as sio
import numpy as np
import torch.optim as optim
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
import mne
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.gridspec as gridspec
from InterpretableCNN import InterpretableCNN

torch.cuda.empty_cache()
torch.manual_seed(0)

plt.rcParams.update({'font.size': 14})

class VisTech():
    def __init__(self, model):
        self.model = model
        self.model.eval()
      
    def heatmap_calculation(self,batchInput,sampleidx,state,radius=32):
        """
        This function generates the heatmap with the visualization technique proposed in the paper.
        input:
           batchInput:          all the samples in a batch for classification
           sampleidx:           index of the sample
           subid:               ID of the subject        
           state:               the classified state output by the classifier. 0--alert, 1--drowsy
           radius:              the influential radius of the Gaussian function. Default is 32, which is half the kernel size.
        """
        
        # to get the activations after the 4th layer
        batchActiv1=self.model.pointwise(batchInput) 
        batchActiv2=self.model.depthwise(batchActiv1)         
        batchActiv3=self.model.activ(batchActiv2)  
        batchActiv4=self.model.batchnorm(batchActiv3)     
        
        # to get the parameters of the network
        layer1weights=self.model.pointwise.weight.cpu().detach().numpy().squeeze()
        layer2weights=self.model.depthwise.weight.detach().cpu().numpy().squeeze()         
        layer6weights=torch.transpose(self.model.fc.weight,0,1).cpu().detach().numpy().squeeze()
        
        # to get activations of the sample
        sampleInput=batchInput[sampleidx].cpu().detach().numpy().squeeze()        
        sampleActiv2=batchActiv2[sampleidx].cpu().detach().numpy().squeeze()        
        sampleActiv4=batchActiv4[sampleidx].cpu().detach().numpy().squeeze()
        
        # to get the dimensional of the input sample and the kernel length of the second layer
        sampleChannel=sampleInput.shape[0] 
        sampleLength=sampleInput.shape[1]
        kernelLength=sampleLength-sampleActiv4.shape[1]+1

        #the class activation map
        CAM=sampleActiv4*np.tile(np.array([layer6weights[:,state]]).transpose(),(1,sampleActiv4.shape[1]))
        CAMsorted=np.sort(CAM,axis=None)
        
        #threshold of the class activation map
        CAMthres=CAMsorted[-100]
        
        # the fixationmap regesters locations of the discriminative points
        fixationmap=CAM>CAMthres
        
        #the activations after 2nd layer and those after 4th layer must have the same signs. Otherwise, you need to change the CAM threshold.
        for i in range(fixationmap.shape[0]):
            for j in range(fixationmap.shape[1]):
                if fixationmap[i,j]:
                    if sampleActiv4[i,j]*sampleActiv2[i,j]<0:
                        fixationmap[i,j]=False
                        print('error')
              
        # the corresponding discriminative locations for the input sample
        fixationmap0=np.zeros((sampleChannel,sampleLength)) 

        # find the corresponding discriminative locations for the input sample
        for i in range(fixationmap.shape[0]):
            for j in range(fixationmap.shape[1]): 
                if fixationmap[i,j]:
                    #Implement equation (13) in the paper
                    sumvalue=np.sum(sampleInput[:,j:j+64]*np.tile(layer2weights[i,:],(sampleChannel,1)),axis=1)*layer1weights[int(np.floor(i/2)),:]
                    # implement equation (14) in the paper
                    p=np.argmax(sumvalue*np.sign(sampleActiv2[i,j]))
                    # implement equation (15) in the paper
                    q=j+int(kernelLength/2)
                    
                    fixationmap0[p,q]=1
          
        # calculate the final heatmap    
        heatmap=np.zeros((sampleChannel,sampleLength))             
        for p in range(sampleChannel):
            for q in range(sampleLength):
                if fixationmap0[p,q]>0:
                                        
                    minbound=int(q-radius)
                    if minbound<0:
                        minbound=0
                        
                    maxbound=int(q+radius)
                    if maxbound>sampleLength:
                        maxbound=sampleLength
                    
                    for qk in range(minbound,maxbound):
                        # implement equation (12) in the paper
                        heatmap[p,qk]=heatmap[p,qk]+ 1 / radius/np.sqrt(2 * np.pi) *np.exp(-(qk-q)** 2/(2*radius*radius))
        
        # normalize the heatmap for visualization
        heatmap= (heatmap-np.mean(heatmap)) / np.sqrt(np.sum(heatmap**2)/(sampleChannel*sampleLength))  
                
        return heatmap   
               
        
              
    def generate_heatmap(self, batchInput,sampleidx,subid,samplelabel,likelihood):
        """
        This function generates figures shown in the figure        
        input:
           batchInput:          all the samples in a batch for classification
           sampleidx:           the index of the sample
           subid:               the ID of the subject
           samplelabel:         the ground truth label of the sample
           likelihood:          the likelihood of the sample to be classified into alert and drowsy state 
        """        
        
        if likelihood[0]>likelihood[1]:
            state=0
        else:
            state=1

        if samplelabel==0:
            labelstr='alert'
        else:
            labelstr='drowsy'        
        
        
        sampleInput=batchInput[sampleidx].cpu().detach().numpy().squeeze()
        sampleChannel=sampleInput.shape[0] 
        sampleLength=sampleInput.shape[1]        
        
        
        heatmap=self.heatmap_calculation(batchInput=batchInput,sampleidx=sampleidx,state=state)
        
        fig = plt.figure(figsize=(23,6))
        
        gridlayout = gridspec.GridSpec(ncols=6, nrows=2, figure=fig,wspace=0.05, hspace=0.005)

        axs0 = fig.add_subplot(gridlayout[0:2,0:2])
        axs1 = fig.add_subplot(gridlayout[0:2,2:4])  
        axs21= fig.add_subplot(gridlayout[0,4]) 
        axs22= fig.add_subplot(gridlayout[0,5]) 
        axs23= fig.add_subplot(gridlayout[1,4]) 
        axs24= fig.add_subplot(gridlayout[1,5])        

    
        fig.suptitle('Subject:'+str(int(subid))+'   '+'Label:'+labelstr+'   '+'$P_{alert}=$'+str(round(likelihood[0],2))+'   $P_{drowsy}=$'+str(round(likelihood[1],2)),y=1.02)  
        thespan=np.percentile(sampleInput,98)        
        
        xx=np.arange(1,sampleLength+1)             
        for i in range(0,sampleChannel):            
            y=sampleInput[i,:]+thespan*(sampleChannel-1-i)
            dydx=heatmap[i,:]           
          
            points = np.array([xx, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(-1, 1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(dydx)
            lc.set_linewidth(2)
            axs0.add_collection(lc)
        
        
        yttics=np.zeros(sampleChannel)
        for gi in range(sampleChannel):
            yttics[gi]=gi*thespan

        axs0.set_ylim([-thespan,thespan*sampleChannel])          
        axs0.set_xlim([0,sampleLength+1]) 
        axs0.set_xticks([1,100,200,300,384])        
        
        channelnames=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8','T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz','O2']
 
        inversechannelnames=[]
        for i in range(sampleChannel):
            inversechannelnames.append(channelnames[sampleChannel-1-i])
                   
        plt.sca(axs0)
        plt.yticks(yttics, inversechannelnames)        
        
        deltapower=np.zeros(sampleChannel)
        thetapower=np.zeros(sampleChannel)
        alphapower=np.zeros(sampleChannel)
        betapower=np.zeros(sampleChannel)
        
        for kk in range(sampleChannel):
            psd, freqs = psd_array_multitaper(sampleInput[kk,:], 128, adaptive=True,normalization='full', verbose=0)
            freq_res = freqs[1] - freqs[0]

            totalpower=simps(psd, dx=freq_res)
            if totalpower<0.00000001:
               deltapower[kk]=0
               thetapower[kk]=0
               alphapower[kk]=0
               betapower[kk]=0
            else:
                idx_band = np.logical_and(freqs >= 1, freqs <= 4)
                deltapower[kk] = simps(psd[idx_band], dx=freq_res)/totalpower
                idx_band = np.logical_and(freqs >= 4, freqs <= 8)
                thetapower[kk]  = simps(psd[idx_band], dx=freq_res)/totalpower
                idx_band = np.logical_and(freqs >= 8, freqs <= 12)
                alphapower[kk]  = simps(psd[idx_band], dx=freq_res)/totalpower       
                idx_band = np.logical_and(freqs >= 12, freqs <= 30)
                betapower[kk]  = simps(psd[idx_band], dx=freq_res)/totalpower

        axs21.set_title('Delta',y=-0.2)
        axs22.set_title('Theta',y=-0.2)
        axs23.set_title('Alpha',y=-0.2)
        axs24.set_title('Beta',y=-0.2)

        montage ='standard_1020'
        sfreq = 128
        
        ch_names=channelnames
        
        info = mne.create_info(
            channelnames,
            ch_types=['eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg',\
                      'eeg', 'eeg', 'eeg', 'eeg', 'eeg'],
            sfreq=sfreq,          
            montage=montage
        )


        topoHeatmap = np.mean(heatmap, axis=1)
        im,cn=mne.viz.plot_topomap(data=topoHeatmap,pos=info, vmin=-1, vmax=1, axes=axs1, names=ch_names,show_names=True,outlines='head',cmap='viridis',show=False)
        fig.colorbar(im,ax=axs1)

        mixpower=np.zeros((4,sampleChannel))
        mixpower[0,:]=deltapower
        mixpower[1,:]=thetapower
        mixpower[2,:]=alphapower
        mixpower[3,:]=betapower

        vmax=np.percentile(mixpower,95)        
        
        im,cn=mne.viz.plot_topomap(data=deltapower,pos=info, vmin=0, vmax=vmax, axes=axs21, names=ch_names,show_names=True,outlines='head',cmap='viridis',show=False)
        im,cn=mne.viz.plot_topomap(data=thetapower,pos=info, vmin=0, vmax=vmax, axes=axs22, names=ch_names,show_names=True,outlines='head',cmap='viridis',show=False)
        im,cn=mne.viz.plot_topomap(data=alphapower,pos=info, vmin=0, vmax=vmax, axes=axs23, names=ch_names,show_names=True,outlines='head',cmap='viridis',show=False)
        im,cn=mne.viz.plot_topomap(data=betapower,pos=info, vmin=0, vmax=vmax, axes=axs24, names=ch_names,show_names=True,outlines='head',cmap='viridis',show=False)
   
        fig.colorbar(im,ax=[axs21,axs22,axs23,axs24])
       

def run():

    lr = 1e-3  
    filename = r'dataset.mat'  
 
    channelnum=30
    classes=2
    subjnum=11
    samplelength=3
   
    tmp = sio.loadmat(filename)
    xdata=np.array(tmp['EEGsample'])
    label=np.array(tmp['substate'])
    subIdx=np.array(tmp['subindex'])

    label.astype(int)    
    subIdx.astype(int)

    samplenum=label.shape[0]
    sf=128
    ydata=np.zeros(samplenum,dtype=np.longlong)
    
    for i in range(samplenum):
        ydata[i]=label[i]

    batch_size = 50   
    n_epoch =11    

    for i in range(1,2):
        trainindx=np.where(subIdx != i)[0] 
        xtrain=xdata[trainindx]   
        x_train = xtrain.reshape(xtrain.shape[0],1,channelnum, samplelength*sf)
      
        y_train=ydata[trainindx]               
        testindx=np.where(subIdx == i)[0]    
        
        y_test=ydata[testindx]
    
        train = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        my_net = InterpretableCNN().double().cuda()
        optimizer = optim.Adam(my_net.parameters(), lr=lr)    
        loss_class = torch.nn.NLLLoss().cuda()

        for p in my_net.parameters():
            p.requires_grad = True    
        
        for epoch in range(n_epoch):  
            for j, data in enumerate(train_loader, 0):
                inputs, labels = data                
                
                input_data = inputs.cuda()
                class_label = labels.cuda()              
           
                my_net.train()          
                my_net.zero_grad()    
                class_output= my_net(input_data) 
                err_s_label = loss_class(class_output, class_label)
                err = err_s_label 
  
                err.backward()
                optimizer.step()

        my_net.eval()
        with torch.no_grad():
            xtest=xdata[testindx]
            x_test = xtest.reshape(xtest.shape[0], 1,channelnum, samplelength*sf)
            x_test =  torch.DoubleTensor(x_test).cuda()
            answer = my_net(x_test)
             
            probs=np.exp(answer.cpu().numpy())
            sampleVis =VisTech(my_net)

            # you can change the sample you want to visualize here
            sampleidx=61
            sampleVis.generate_heatmap(batchInput=x_test,sampleidx=sampleidx,subid=i,samplelabel=y_test[sampleidx],likelihood=probs[sampleidx])

if __name__ == '__main__':
    run()
    
