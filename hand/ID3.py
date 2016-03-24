# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:09:51 2016

@author: yiyuezhuo
"""

'''
It implement Machine Learning by Tom Mitchell Chapter 3 ID3 algorithm
'''

import pandas as pd
import numpy as np


def plogp(p):
    return p*np.log2(p) if p!=0 else 0

def Entropy(S):
    '''
    :param S: Series object
    '''
    count=S.groupby(S).count()
    pt=count/count.sum()
    return -pt.map(plogp).sum()
    
def Gain(S,A):
    '''
    :param S: Series object,Target_attribute series of examples dataframe
    :param A: Series object,one Attribute series of examples dataframe,
                it is indicated by a element in Attributes
    '''
    entropy_past=Entropy(S)
    g=S.groupby(A)
    p=g.count()/g.count().sum()
    entropy_now=(g.agg(Entropy)*p).sum()
    return entropy_past-entropy_now

class Node(object):
    def __init__(self,examples,Target_attribute,Attributes):
        '''
        :param examples: DataFrame object.
        :param Target_attribute: string to indicate the examples column
        :param Attributes: string list to indicate the examples columns
        '''
        self.examples=examples
        self.Target_attribute=Target_attribute
        self.Attributes=Attributes
        
        self.child={}
        self.default=None
        self.check=None
    def build(self):
        examples=self.examples
        Target_attribute=self.Target_attribute
        Attributes=self.Attributes
        groupby=examples[Target_attribute].groupby(examples[Target_attribute])
        self.default=groupby.count().argmax()
        if len(self.Attributes)==0 or groupby.count().count()==1:
            return
        else:
            gain_t=examples[Attributes].apply(lambda x:Gain(x,examples[Target_attribute]))
            self.check=gain_t.argmax()
            sub_examples=dict(list(examples.drop(self.check,1).groupby(examples[self.check])))
            s_Attributes=Attributes[:]
            s_Attributes.remove(self.check)
            for skey,s_examples in sub_examples.items():
                node=Node(s_examples,Target_attribute,s_Attributes)
                self.child[skey]=node
                node.build()
    def predict(self,record):
        if len(self.child)==0:
            return self.default
        elif not self.child.has_key(record[self.check]):
            return self.default
        return self.child[record[self.check]].predict(record)
    def accuracy(self,data=None,Target_attribute=None):
        if not Target_attribute:
            Target_attribute=self.Target_attribute
        if not data:
            data=self.examples
        predicted=data.apply(self.predict,axis=1)
        df=data[Target_attribute]==predicted
        return df.mean()

if __name__=='__main__':
    data=pd.read_csv('../data/ID3test.csv')
    tree=Node(data,'PlayTennis',['Outlook','Humidity','Wind','Temperature'])
    tree.build()
    
    print tree.predict(data.ix[0])
    print tree.accuracy()