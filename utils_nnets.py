import torch
import torch.nn as nn

from . import config


#base class for all the models to allow for dynamic resize of tensor for noises
class MamasitaNetwork(nn.Module):
	#utils
	dynamicgeneration=[0]
	dynamicbatch=[0]

#class for gaussian addition in a sequencial
class add_gaussian(MamasitaNetwork):
	def __init__(self,shape,std=0.0):
		super(add_gaussian, self).__init_q_()
		self.std=std
		self.conv=False
		if len(shape)==2:
			self.a,self.b=shape
		if len(shape)==4:
			self.a,self.b,self.c,self.d=shape
			self.conv=True
		if std!=0:
			self.sampler=torch.zeros(shape).cuda()

	def forward(self,x):
		if self.training and self.std!=0.0 and not self.dynamicgeneration[0]:
			x+=self.sampler.normal_(0,self.std)

		if self.training and self.std!=0.0 and self.dynamicgeneration[0]:
			if self.conv:
				aux=torch.zeros(self.dynamicbatch[0],self.b,self.c,self.d).cuda().normal_(0,self.std)
			else:
				aux=torch.zeros(self.dynamicbatch[0],self.b).cuda().normal_(0,self.std)
			x.data+=aux
		return x

#class for linear activation
class Linear_act(nn.Module):
	def __init__(self):
		super(Linear_act, self).__init__()
	def forward(self,x):
		return x

def linear(x):
	return x

def return_activation(act,dim=1):
	if act=='relu':
		return nn.ReLU()
	elif act=='linear':
		return Linear_act()
	elif act=='softmax':
		return nn.Softmax(dim)
	elif act=='sigmoid':
		return nn.Sigmoid()
	elif act=='tanh':
		return nn.Tanh()
	else:
		raise NotImplemented

def apply_DeConv(inp,out,kernel,act,shape=None,std=0.0,drop=0.0,bn=True,stride=1,padding=1,output_padding=0):
	bias = False if bn else True

	conv=nn.ConvTranspose2d(inp,out, kernel,stride=stride, padding=padding,bias=bias,output_padding=output_padding)

	if bn:
		convBN=nn.BatchNorm2d(out)
	activation=return_activation(act)

	if drop!=0:
		dropl=nn.Dropout2d(drop)

	if std!=0.0:
		assert shape!=None
		nlayer=add_gaussian(shape,std)

	#define the sequential
	forward_list=[conv]
	if bn:
		forward_list.append(convBN)
	if std!=0.0:
		forward_list.append(nlayer)

	forward_list.append(activation)
	if drop!=0:
		forward_list.append(dropl)

	return nn.Sequential(*forward_list)


def apply_conv(inp,out,kernel,act,shape=None,std=0.0,drop=0.0,bn=True,stride=1,padding=1):
	bias = False if bn else True

	conv=nn.Conv2d(inp,out,kernel,bias=bias,padding=padding,stride=stride)

	if bn:
		convBN=nn.BatchNorm2d(out)

	activation=return_activation(act)

	if drop!=0:
		dropl=nn.Dropout2d(drop)
	if std!=0.0:
		assert shape!=None
		nlayer=add_gaussian(shape,std)

	#define the sequential
	forward_list=[conv]
	if bn:
		forward_list.append(convBN)

	if std!=0.0:
		forward_list.append(nlayer)

	forward_list.append(activation)

	if drop!=0:
		forward_list.append(dropl)

	return nn.Sequential(*forward_list)

def apply_pool(kernel):
	mp=nn.MaxPool2d(kernel)
	return mp

def apply_DePool(kernel):
	mp=nn.UpsamplingBilinear2d(kernel)
	return mp

def apply_linear(inp,out,act,shape=None,std=0.0,drop=0.0,bn=True):
	bias = False if bn else True

	w=nn.Linear(inp,out,bias=bias)
	if bn:
		wBN=nn.BatchNorm1d(out)
	activation=return_activation(act)
	if drop!=0:
		dropl=nn.Dropout(drop)
	if std!=0.0:
		assert shape!=None
		nlayer=add_gaussian(shape,std)

	#define the sequential
	forward_list=[w]
	if bn:
		forward_list.append(wBN)

	if std!=0.0:
		forward_list.append(nlayer)

	forward_list.append(activation)

	if drop!=0:
		forward_list.append(dropl)

	return nn.Sequential(*forward_list)



def categorical_to_one_hot(t,max_val):
	one_hot = torch.zeros(t.size(0),max_val, device=t.device)
	one_hot.scatter_(1,t.view(-1,1),1)
	return one_hot
