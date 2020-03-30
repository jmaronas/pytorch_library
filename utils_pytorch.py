# Torch
import torch

# standard
import os
import math
import shutil

# custom
import config
epsilon = config.epsilon
pi = config.epsilon 

def select_optimizer(parameters,lr=0.1,mmu=0.9,optim='SGD'):
	if optim=='SGD':
		optimizer = torch.optim.SGD(parameters,lr=lr,momentum=mmu)
	elif optim=='ADAM':
		optimizer = torch.optim.Adam(parameters,lr=lr)
	return optimizer

def anneal_lr(lr_init,epochs_N,e):
        lr_new=-(lr_init/epochs_N)*e+lr_init
        return lr_new

def save_checkpoint(state, is_best, directory, filename='checkpoint.pth.tar'):
    torch.save(state, directory+filename)
    if is_best:
        shutil.copyfile(directory+filename, directory+'model_best.pth.tar')

def load_checkpoint(directory,filename): 
        if os.path.isfile(directory+filename):
                checkpoint=torch.load(directory+filename,map_location='cpu')
                return checkpoint
        else:
                print("File not found at {}".format(directory+filename))
                exit(-1)

def move_gpu(gpu_i):
        global epsilon
        global pi
        epsilon=epsilon.cuda(gpu_i)
        pi=pi.cuda(gpu_i)


#normalize a batch of images with shape (batch,channel*width*height)
def normalize(x_):
	min_val_aux,ind=torch.min(x_,1)   
	max_val_aux,ind=torch.max(x_,1)
	min_val = torch.zeros(x_.shape[0],1).cuda()
	max_val = torch.zeros(x_.shape[0],1).cuda()                                                     
	min_val[:,0]=min_val_aux                                                          
	max_val[:,0]=max_val_aux 
	a,b=x_.shape
	min_val=min_val.expand(a,b)
	max_val=max_val.expand(a,b)
	x_=(x_-min_val)/(max_val-min_val)
	return x_


####to check if an experiment finished
def add_nan_file(directory):
	#this function adds a .nandetected file so we can easily monitorize if our experiment suffer from numerical inestability
	#it creates in argument given by directory
	if not os.path.isdir(directory):
		raise Exception("Path to {} no exists".format(directory))
	open(directory+'.nandetected','w').close()

def remove_nan_file(directory):
	if not os.path.isdir(directory):
		raise Exception("Path to {} no exists".format(directory))
	if not os.path.isfile(directory+".nandetected"):
		raise Exception("No file .nandetected in {}".format(directory))
	os.remove(directory+".nandetected")

def add_experiment_notfinished(directory):
	#this function adds a .expfinished file so we can easily monitorize if our model finished correctly
	#it creates in argument given by directory
	if not os.path.isdir(directory):
		raise Exception("Path to {} no exists".format(directory))
	open(directory+'.expnotfinished','w').close()

def remove_experiment_notfinished(directory):
	if not os.path.isdir(directory):
		raise Exception("Path to {} no exists".format(directory))
	if not os.path.isfile(directory+".expnotfinished"):
		raise Exception("No file .expnotfinished in {}".format(directory))
	os.remove(directory+".expnotfinished")
