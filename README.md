# EEG-based-Cross-Subject-Driver-Drowsiness-Recognition-with-an-Interpretable-CNN
Pytorch implementation of the model "InterpretableCNN" proposed in the paper "EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network".

If you find the codes useful, pls cite the paper:

J. Cui, Z. Lan, O. Sourina and W. Müller-Wittig, "EEG-Based Cross-Subject Driver Drowsiness Recognition With an Interpretable Convolutional Neural Network," in IEEE Transactions on Neural Networks and Learning Systems, doi: 10.1109/TNNLS.2022.3147208.

The project contains 3 code files. They are implemented with Python 3.6.6.

"InterpretableCNN.py" contains the model. required library: torch

"LeaveOneOut_acc.py" contains the leave-one-subject-out method to get the classifcation accuracies. It requires the computer to have cuda supported GPU installed. required library:torch,scipy,numpy,sklearn

"VisTechnique.py" contains the novel visualization technique proposed in the paper. It requires the computer to have cuda supported GPU installed. required library:torch,scipy,numpy,matplotlib,mne

The processed dataset has been uploaded to: https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687

If you have any problems, please Contact Dr. Cui Jian at cuij0006@ntu.edu.sg
