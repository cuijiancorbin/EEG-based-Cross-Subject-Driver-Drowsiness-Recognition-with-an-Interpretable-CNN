# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 18:17:29 2019

@author: JIAN
"""
import torch

class InterpretableCNN(torch.nn.Module):  
    
    """
    The codes implement the CNN model proposed in the paper "EEG-based Cross-Subject Driver Drowsiness Recognition
    with an Interpretable Convolutional Neural Network".(doi: 10.1109/TNNLS.2022.3147208)
	
    The network is designed to classify multi-channel EEG signals for the purposed of driver drowsiness recognition.
    
    Parameters:
        
    classes       : number of classes to classify, the default number is 2 corresponding to the 'alert' and 'drowsy' labels.
    sampleChannel : channel number of the input signals.
    sampleLength  : the length of the EEG signals. The default value is 384, which is 3s signal with sampling rate of 128Hz.
    N1            : number of nodes in the first pointwise layer.
    d             : number of kernels for each new signal in the second depthwise layer.      
    kernelLength  : length of convolutional kernel in second depthwise layer.
   
    if you have any problems with the code, please contact Dr. Cui Jian at cuij0006@ntu.edu.sg
    """    
    
    def __init__(self, classes=2, sampleChannel=30, sampleLength=384 ,N1=16, d=2,kernelLength=64):
        super(InterpretableCNN, self).__init__()
        self.pointwise = torch.nn.Conv2d(1,N1,(sampleChannel,1))
        self.depthwise = torch.nn.Conv2d(N1,d*N1,(1,kernelLength),groups=N1) 
        self.activ=torch.nn.ReLU()       
        self.batchnorm = torch.nn.BatchNorm2d(d*N1,track_running_stats=False)       
        self.GAP=torch.nn.AvgPool2d((1, sampleLength-kernelLength+1))         
        self.fc = torch.nn.Linear(d*N1, classes)        
        self.softmax=torch.nn.LogSoftmax(dim=1)

    def forward(self, inputdata):
        intermediate = self.pointwise(inputdata)        
        intermediate = self.depthwise(intermediate) 
        intermediate = self.activ(intermediate) 
        intermediate = self.batchnorm(intermediate)          
        intermediate = self.GAP(intermediate)     
        intermediate = intermediate.view(intermediate.size()[0], -1) 
        intermediate = self.fc(intermediate)    
        output = self.softmax(intermediate)   

        return output  
