# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 07:48:37 2016

@author: yiyuezhuo
"""

import numpy as np

def p_one(x):
    y=np.zeros(x.shape)
    for index,loc in enumerate(np.argmax(x,axis=0)):
        y[loc,index]=1.0
    return y

def h_theta(x,theta):
    h=np.exp(np.dot(theta,x))
    return h/np.sum(h,axis=0)
    
def J(x,y,theta):
    '''loss function'''
    m=x.shape[1]
    return (-1.0/m)*np.sum(y*np.log(h_theta(x,theta)))
    
def grad_theta_J(x,y,theta):
    ''' x col is a sample feature
    y is dummy matrix its col is dummy vector 
    the one loc is its classification else are zero'''
    m=x.shape[1]
    p=h_theta(x,theta)
    return (-1.0/m)*np.dot(x,(y-p).T).T
    
def grad_descent(x,y,theta,alpha=0.1,iters=1000):
    for i in range(iters):
        grad=grad_theta_J(x,y,theta)
        theta-=alpha*grad
    return theta
    
class Softmax(object):
    def __init__(self,x,y,init_theta=None):
        self.x=x
        self.y=y
        self.theta=init_theta
        if self.theta==None:
            self.theta=np.random.random((self.y.shape[0],self.x.shape[0]))
    def train(self,alpha=0.1,iters=1000):
        #grad_descent(self.x,self.y,self.theta,alpha=alpha,iters=iters)
        x,y,theta=self.x,self.y,self.theta
        for i in range(iters):
            #print 'iter',i,'loss',self.loss(),'accuracy',self.accuracy()
            print('iter {} loss {:.3f} accuracy {:.3f}'.format(i,self.loss(),self.accuracy()))
            grad=grad_theta_J(x,y,theta)
            theta-=alpha*grad
        self.theta=theta
    def predict(self,x):
        return h_theta(self.x,self.theta)
    def loss(self):
        return J(self.x,self.y,self.theta)
    def accuracy(self):
        sy=p_one(self.predict(self.x))
        return 1-np.sum(sy!=self.y)/(sy.shape[1]*2.0)

if __name__=='__main__':
    feature=np.array([[1,1,1],[1,1,1],[0,1,1],[0,0,1]]).T
    #feature=np.array([[1,1],[1,0],[0,1],[0,0]]).T
    target=np.array([[0,1],[1,0],[1,0],[0,1]]).T
    sf=Softmax(feature,target)
    sf.train(alpha=0.1,iters=100)
    print sf.predict(sf.x)
    '''
    [[ 0.53702532  0.53702532  0.68309672  0.40599736]
     [ 0.46297468  0.46297468  0.31690328  0.59400264]]
    However Softmax use linear combinations as base so it can't
    identify XOR clearly.In fact, if you use XOR origin form,you will
    get a ~0.5 accuracy and result is very close to 0.5-0.5
    '''
    
    x=feature
    y=target
    theta=sf.theta