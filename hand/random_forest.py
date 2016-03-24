# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 17:08:34 2016

@author: yiyuezhuo
"""

import numpy as np
import pandas as pd
import random
import ID3
from collections import Counter

def sample(examples,Target_attribute,Attributes,row_number,col_number):
    drop_Attributes=random.sample(Attributes,len(Attributes)-col_number)
    r_examples=examples.drop(drop_Attributes,axis=1).sample(row_number,replace=True)
    r_Attributes=list(set(Attributes)-set(drop_Attributes))
    return r_examples,Target_attribute,r_Attributes
    #return df[random.sample(df.columns,col_number)].sample(row_number,replace=True)

class RandomForest(object):
    def __init__(self,examples,Target_attribute,Attributes,
                 attribute_number=None,tree_number=500):
        self.examples=examples
        self.Target_attribute=Target_attribute
        self.Attributes=Attributes
        self.tree_number=tree_number

        if not attribute_number:
            attribute_number=int(np.sqrt(len(Attributes)))+1
            self.attribute_number=attribute_number
        self.tree_list=[]
        count=len(examples)
        for i in range(tree_number):
            #exam=sample(examples,count,attribute_number)
            tree=ID3.Node(*sample(examples,Target_attribute,Attributes,count,attribute_number))
            #tree=ID3.Node(exam,Target_attribute,Attributes)
            self.tree_list.append(tree)
    def build(self):
        for tree in self.tree_list:
            tree.build()
    def predict(self,record,order=False):
        ct=Counter([tree.predict(record) for tree in self.tree_list])
        if order:
            return ct
        return ct.most_common(1)[0][0]
    def out_of_bag_error(self,examples=None):
        # use tree can't use data to test
        if type(examples)==type(None):
            examples=self.examples
        wrong_l=[]
        for key,record in examples.iterrows():
            key_result=[]
            for tree in self.tree_list:
                if key not in tree.examples.index:
                    key_result.append(tree.predict(record))
            ob_predict=Counter(key_result).most_common(1)[0][0]
            origin_value=record[self.Target_attribute]
            #print 'compare',ob_predict,origin_value
            if ob_predict!=origin_value:
                wrong_l.append((key,ob_predict,origin_value))
        return float(len(wrong_l))/len(self.examples)
    def variable_importance(self):
        # shuffle one examples columns to evalate it
        mapping={}
        origin=self.out_of_bag_error()
        for Attributes_test in self.Attributes:
            examples=self.examples.copy()
            new_col=pd.Series(random.sample(examples[Attributes_test],len(examples)),index=examples.index)
            examples[Attributes_test]=new_col
            value=self.out_of_bag_error(examples)-origin
            mapping[Attributes_test]=value
        return mapping
        
if __name__=='__main__':
    data=pd.read_csv('../data/ID3test.csv')
    rf=RandomForest(data,'PlayTennis',['Outlook','Humidity','Wind','Temperature'],tree_number=100)
    rf.build()
    print 'out of bag error',rf.out_of_bag_error()
    print 'variable_importance',rf.variable_importance()

