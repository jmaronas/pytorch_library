import torch
if torch.__version__ != '1.0.0':
	raise RuntimeError('PyTorch version must be 1.0.0')

import torch.nn.functional as F
import math

epsilon=(torch.ones(1,)*1e-11)
pi=(torch.ones(1,)*float(math.pi))

#computes the gaussian kullback lieber divergence of two matrix representing (batch,dimensions)
def DKL_gaussian(mean_q,logvar_q,mean_p,logvar_p,reduce_batch_dim=False,reduce_sample_dim=False):
	#computes the DKL(q(x)//p(x)) per gaussian of each sample in a batch. Returns (batch,DKL)
	var_p = torch.exp(logvar_p)
	var_q = torch.exp(logvar_q)
	DKL=0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p)

	if reduce_sample_dim and reduce_batch_dim:
		return DKL.sum()		

	if reduce_sample_dim:
		return DKL.sum(1)		
	return DKL

#same as above but usefull for bayesian neural networks
def DKL_gaussian_optimized(mean_q,logvar_q,mean_p,logvar_p,reduce_batch_dim=False,reduce_sample_dim=False):
	'''
	Computes the DKL(q(x)//p(x)) between to factorized gaussian distributions q(x) and p(x)
		mean_q->mean of qx distribution
		logvar_q->log variance of qx distribution
		mean_p->mean of px distribution
		logvar_p->logvariance of px distribution
	'''
	alogvar_q=logvar_q
	amean_q=mean_q 
	var_p = torch.exp(logvar_p)

	if reduce_batch_dim and reduce_sample_dim:
		DKL=0.0
		for mean_q,logvar_q in zip(amean_q,alogvar_q):
			var_q = torch.exp(logvar_q)
			DKL+=0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p).sum()

	elif reduce_sample_dim:
		DKL=[]
		for mean_q,logvar_q in zip(amean_q,alogvar_q):
			var_q = torch.exp(logvar_q)
			DKL.append(0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p)).sum(1)
	else:
		DKL=[]
		for mean_q,logvar_q in zip(amean_q,alogvar_q):
			var_q = torch.exp(logvar_q)
			DKL.append(0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p))		
	return DKL


#same as above but usefull for bayesian neural networks
def DKL_gaussian_optimized_learnprior(mean_q,logvar_q,mean_p,logvar_p,reduce_batch_dim=False,reduce_sample_dim=False):
        alogvar_q=logvar_q
        amean_q=mean_q

        alogvar_p=logvar_p
        amean_p=mean_p

        if reduce_batch_dim and reduce_sample_dim:
                DKL=0.0
                for mean_q,logvar_q,mean_p,logvar_p in zip(amean_q,alogvar_q,amean_p,alogvar_p):
                        var_q = torch.exp(logvar_q)
                        var_p = torch.exp(logvar_p)
                        DKL+=0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p).sum()

        elif reduce_sample_dim:
                DKL=[]
                for mean_q,logvar_q,mean_p,logvar_p in zip(amean_q,alogvar_q,amean_p,alogvar_p):
                        var_q = torch.exp(logvar_q)
                        var_p = torch.exp(logvar_p)
                        DKL.append(0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p)).sum(1)
        else:
                DKL=[]
                for mean_q,logvar_q in zip(amean_q,alogvar_q):
                        var_q = torch.exp(logvar_q)
                        var_p = torch.exp(logvar_p)
                        DKL.append(0.5 * (-1 + logvar_p - logvar_q + (var_q/var_p) + torch.pow(mean_p-mean_q,2)/var_p))
        return DKL


def Entropy_gaussian_optimized(mean_p,logvar_p):
	'''
	Computes the differential entropy of a gaussian distribution with given mean_p and logvar_p
	'''
	global pi
	alogvar=logvar_p
	amean=mean_p
	entropy=0.0
	for mean,logvar in zip(amean,alogvar):
		var=torch.exp(logvar)
		entropy += 0.5*(torch.log(var) + torch.log(2*pi) + 1)

	return entropy

def Entropy_gaussian(mean_p,logvar_p):
	'''
	Computes the differential entropy of a gaussian distribution with given mean_p and logvar_p
	'''
	global pi
	var=torch.exp(logvar_p)
	entropy = 0.5*(torch.log(var) + torch.log(2*pi) + 1)
	return entropy

def Entropy_categorical(p):
	'''
	Computes the differential entropy of a categorical distribution
	'''
	return (-1*p*torch.log(p)).sum(1)


def sampler(param,shape,return_mean=False,distribution='gauss'):
	#sample from a gaussian distribution of a given shape
	if distribution=='gauss':
		mean,logvar=param
		if return_mean:
			return mean

		std = torch.exp(logvar*0.5)
		sample=torch.zeros(shape).cuda().normal_(0,1)
		sample=sample*std+mean
	elif distribution=='bern':
		if return_mean:
			mean,logvar=param
			return mean
		else:
			mean,logvar=param
			sample=torch.bernoulli(mean).cuda()		
	else:
		raise NotImplemented

	return sample

#this version of sampling is usefull for bayesian neural networks
def sampler_optimized(param,monte_carlo_sampling_list):
	mean,logvar=param
	samples_list=list()
	for index in range(len(monte_carlo_sampling_list)):
		m,logv=mean[index],logvar[index]
		std = torch.exp(logv*0.5)
		monte_carlo_sampling_list[index].normal_(0,1)
		samples_list.append(Variable(monte_carlo_sampling_list[index])*std+m)
			
	return samples_list



#computes log-likelihood of a distribution when the likelihood of such distribution is not a typical cost function such as sse, [binary/categorical]crossentropy
def likelihood(parameters,x,tipe='gauss',reduce_batch_dim=False,reduce_sample_dim=False):	
	if tipe=='gauss': 
		mean,logvar=parameters
		var = torch.exp(logvar)
		std = torch.exp(logvar*0.5)
		lle=-1/(2*var)*torch.pow(x-mean,2)-torch.log(epsilon+std)-0.5*torch.log(epsilon+2*pi)
	elif tipe=='bern':
		
		mean,_=parameters
		lle=-1*F.binary_cross_entropy(mean,x,size_average=False,reduce=False)
	
	else:
		raise NotImplemented
	if reduce_batch_dim and reduce_sample_dim:
		return lle.sum()

	if reduce_sample_dim:
		return lle.sum(1)

	return lle

###################
#TODO check this with the original code from the folder DA_MCMC
#to sample from a variatonal autoencoder either by sampling a latent variable from the prior distribution or by running a markov chain starting from an observed image. 
'''
Not erase but unseful
def vae_sample_images(vae_net,n_samples,x=None,return_mean=False):
	
	prior_samples=vae_net.get_sample_from_prior(n_samples,return_mean=return_mean)
	if return_mean:
		prior_samples=prior_samples.data

	posterior_samples=list()
	if x!=None:
		for x_ in x:
			if return_mean:
				posterior_samples.append(vae_net.get_sample_from_posterior(x_,return_mean=return_mean).data)
			else:
				posterior_samples.append(vae_net.get_sample_from_posterior(x_,return_mean=return_mean))

		return prior_samples,posterior_samples

	return prior_samples
'''
