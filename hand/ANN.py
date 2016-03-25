# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 17:28:55 2016

@author: yiyuezhuo
"""

import numpy as np


def sigmoid(x):
    return 1.0/(1+np.exp(-x))
    
def d_sigmoid(x):
    return x*(1-x)
    

class Layer(object):
    def __init__(self,input_size,neuron_size,no_linear='sigmoid'):
        f_mapping={'sigmoid':sigmoid}
        df_mapping={'sigmoid':d_sigmoid}
        self.forward_f=f_mapping[no_linear]
        self.backward_f=df_mapping[no_linear]
        self.W=np.random.random((neuron_size,input_size))
        self.X=None
        self.O=None
    def forward(self,X):
        self.X=X
        self.O=self.forward_f(np.dot(self.W,X))
        return self.O
    def backward(self):
        pass
    
class Dense(object):
    def __init__(self,dim,size,no_linear='sigmoid'):
        '''
        :param dim: single example input vector dimension
        :param size: list like [100,50,10] mean 3 layer with 100,50,10 neuron
        :param no_linear: function used in network for no-linear transform
        '''
        self.layer_list=[Layer(*ins,no_linear=no_linear) for ins in zip([dim]+size[:-1],size)]
    def predict(self,input_vector):
        '''vector can be 2d array or matrix like'''
        vector=input_vector
        for layer in self.layer_list:
            vector=layer.forward(vector)
        return vector
    def output(self,input_vector):
        vector=input_vector
        output_l=[]
        for layer in self.layer_list:
            vector=layer.forward(vector)
            output_l.append(vector)
        return output_l
    def to_train(self,input_vector):
        return zip(self.layer_list,self.output(input_vector))
        

    
class BackPropagation(object):
    def __init__(self,network,feature,target,mu=0.1):
        self.network=network
        self.feature=feature
        self.target=target
        self.mu=mu
        assert self.feature.shape[1]==self.target.shape[1]
    def train_one(self,verbose=1):
        mu=self.mu
        loss_l=[]
        for j in range(self.feature.shape[1]):
            feature=self.feature[:,j]
            target=self.target[:,j]
            to_train=self.network.to_train(feature)
            to_train.reverse()
            delta_l=[]
            # finite output layer special process
            layer,output=to_train[0]
            loss=0.5*np.sum((output-target)**2)
            loss_l.append(loss)
            #print 'sample',j,'loss',loss
            delta=(target-output)*output*(1-output)
            delta_l.append(delta)
            for i in range(1,len(to_train)):
                last_layer,last_output=to_train[i-1]
                layer,output=to_train[i]
                delta=output*(1-output)*np.dot(delta_l[-1],last_layer.W)
                delta_l.append(delta)
            X=list(zip(*to_train)[1][1:])+[feature]
            dw_l=[]
            for x,delta,layer,output in zip(X,delta_l,*zip(*to_train)):
                dw=mu*np.outer(delta,x)# I first use the outer 
                dw_l.append(dw)
                layer.W+=dw
        total_loss=np.sum(loss_l)
        #print 'total sample loss',total_loss
        return total_loss
    def train(self,n,verbose=1):
        loss_l=[]
        for i in range(n):
            if verbose>=1:
                print 'training',i,'epoch'
            loss=self.train_one(verbose=verbose)
            loss_l.append(loss)
            if verbose>=1:
                print 'train loss',loss
        return loss_l

if __name__=='__main__':
    ''' train network to learn XOR function '''
    network=Dense(2,[3,3,1])
    feature=np.array([[1,1],[1,0],[0,1],[0,0]]).T
    target=np.array([[0],[1],[1],[0]]).T
    bp=BackPropagation(network,feature,target,mu=1)
    bp.train(4000,verbose=0)
    print network.predict(feature)#[[ 0.03550935  0.9736272   0.97313466  0.00390236]]
    