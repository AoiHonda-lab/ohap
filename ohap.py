import numpy as np
import os
import pandas as pd
import torch
from scipy.special import comb
from itertools import combinations
import matplotlib.pyplot as plt
import japanize_matplotlib
import copy
import shap

class OHAP:

    #　組み合わせを作る関数
    def power_set(self, columns_num):
        items = [i for i in range(1, columns_num+1)]
        sub_sets=[]
        for i in range(len(items) + 1):
            if i > columns_num:
                break
            for c in combinations(items, i):
                sub_sets.append(c)
        all_sets = sub_sets
        
        return all_sets    
    
    #　組み合わせを作る関数
    def power_set_index(self, columns_num):
        items = [i for i in range(0, columns_num)]
        sub_sets=[]
        for i in range(len(items)+1):
            if i > columns_num:
                break
            for c in combinations(items, i):
                sub_sets.append(c)
        all_sets = sub_sets
        
        return all_sets
   
    def ohap(self, model, X_test, grouping, columns=[]):
        if columns == []:
            self.feature_name = model.feature_names_in_
        else:
            self.feature_name = columns

        s = self.power_set_index(len(X_test[0]))
        sh = torch.zeros([len(X_test), len(s)])
        
        if type(X_test) is type(np.array([0])):
            for i in range(len(X_test)):
                for j in range(len(s)):
                    X_test2=copy.deepcopy(X_test)
                    X_test2[:,s[j]]=X_test2[i,s[j]]
                    sh[i,j] = model.predict(X_test2).mean()    
        else:
            for i in range(len(X_test)):
                for j in range(len(s)):
                    X_test2=X_test.clone()
                    X_test2[:,s[j]]=X_test2[i,s[j]]
                    sh[i,j] = model.predict(X_test2).mean()               
        
        
        all_sets = (self.power_set(len(X_test[0]))) 
        fuzzy_ = []       
        for i in range(0, len(s)):
            f = 0
            f2 = 0
            for j in range(0, len(s)):
                A = all_sets[i]
                B = all_sets[j]
                if  set(B) <= set(A): #"<="は部分集合か、"<"にしたら真部分集合か判定できる
                    f += sh[:,j]*(-1)**(len(all_sets[i])-len(all_sets[j]))
            fuzzy_.append(f)
        d_fuzzy = dict(zip(all_sets, fuzzy_))
        
        
        owen = []          # Owen Value
        g_num = len(grouping) # グループ数
        for i in all_sets[1:len(X_test[0])+1]:
            owen_val = 0
            for S in all_sets:
                count = np.zeros(g_num) # 各グループに含まれる要素数
                i_group = 0 # iが何番目のグループに属するか格納する
                j = 0
                if set(i) <= set(S):
                    for g in grouping:
                        common  =  set(g) & set(S) # 和集合
                        count[j] = len(common)
                        if set(common) >= set(i):
                            i_group = j
                        else:
                            pass
                        j += 1

                    # np.count_nonzero(count)はメンバーがゼロではないグループ数。つまりSのメンバーが属するグループ数
                    # count[i_group]はiが属するグループに属する要素数。iも込み。
                    owen_val += d_fuzzy[S] / (np.count_nonzero(count) * count[i_group])
                else:
                    pass
            owen.append(owen_val)     
        owen2=owen[0].reshape(1,len(X_test))
        for i in range(1,len(X_test[0])):
            l = owen[i].reshape(1,len(X_test))
            owen2= torch.cat((owen2,l),0)
        ohap_value=owen2.T
        
        ohap_value = ohap_value.to('cpu').detach().numpy().copy()
        mean2=sh.to('cpu').detach().numpy().copy()
        mean=mean2[0][0]
        
        self.ohap_value2 = [0]*len(X_test)
        if columns!=[]:
            X_test = X_test.to('cpu').detach().numpy().copy()
        
        for i in range(len(X_test)):
            self.ohap_value2[i]= shap.Explanation(ohap_value[i],mean,X_test[i],feature_names=self.feature_name)
            
        self.ohap_value = ohap_value     
        
        return ohap_value
    

    
    
    def shap(self, model, X_test, columns=[]):
        if columns == []:
            self.feature_name = model.feature_names_in_
        else:
            self.feature_name = columns 

        s = self.power_set_index(len(X_test[0]))
        sh = torch.zeros([len(X_test), len(s)])
        
        if type(X_test) is type(np.array([0])):
            for i in range(len(X_test)):
                for j in range(len(s)):
                    X_test2=copy.deepcopy(X_test)
                    X_test2[:,s[j]]=X_test2[i,s[j]]
                    sh[i,j] = model.predict(X_test2).mean()    
        else:
            for i in range(len(X_test)):
                for j in range(len(s)):
                    X_test2=X_test.clone()
                    X_test2[:,s[j]]=X_test2[i,s[j]]
                    sh[i,j] = model.predict(X_test2).mean()  
                 
                
        all_sets = (self.power_set(len(X_test[0]))) 
        fuzzy_ = []
        for i in range(0, len(s)):
            f = 0
            for j in range(0, len(s)):
                A = all_sets[i]
                B = all_sets[j]
                if  set(B) <= set(A): #"<="は部分集合か、"<"にしたら真部分集合か判定できる
                    f += sh[:,j]*(-1)**(len(all_sets[i])-len(all_sets[j]))
            fuzzy_.append(f)
        d_fuzzy = dict(zip(all_sets, fuzzy_))
        
        
        shapley = []
        for i in range(1, len(X_test[0])+1):
            shapley_val = 0
            A = all_sets[i]
            for B in all_sets:
                if set(A) <= set(B):
                    shapley_val += d_fuzzy[B] / len(B)
            shapley.append(shapley_val) 
            
        shapley2=shapley[0].reshape(1,len(X_test))
        for i in range(1,len(X_test[0])):
            l = shapley[i].reshape(1,len(X_test))
            shapley2= torch.cat((shapley2,l),0)
        shap_value=shapley2.T        
        
        shap_value = shap_value.to('cpu').detach().numpy().copy()
        mean2=sh.to('cpu').detach().numpy().copy()
        mean=mean2[0][0]
        
        self.shap_value2 = [0]*len(X_test)
        if columns!=[]:
            X_test = X_test.to('cpu').detach().numpy().copy()
        
        for i in range(len(X_test)):
            self.shap_value2[i]= shap.Explanation(shap_value[i],mean,X_test[i],feature_names=self.feature_name)
            
        self.shap_value = shap_value    
        
        return shap_value


    
    def waterfall_plot(self, i, value_type='ohap'):
        if value_type=='ohap':
            return shap.plots.waterfall(self.ohap_value2[i])
        else :
            return shap.plots.waterfall(self.shap_value2[i])
        
        
    def bar_plot(self, value_type='ohap'):
        if value_type=='ohap':
            # step1 データの作成
            labels = self.feature_name#['G1', 'G2', 'G3', 'G4', 'G5']
            x = np.arange(len(labels))

            spp=np.mean(abs(self.ohap_value), axis=0)

            # step2 グラフフレームの作成
            fig, ax = plt.subplots()
            # step3 棒グラフの描画
            ax.barh(x, spp, tick_label=labels)
            ax.set_xlabel('average of absolute ohap values')
            ax.set_ylabel('feature names')
            ax.set_title('bar plot')
            plt.show()
            
        else :
            labels = self.feature_name#['G1', 'G2', 'G3', 'G4', 'G5']
            x = np.arange(len(labels))

            spp=np.mean(abs(self.shap_value), axis=0)

            # step2 グラフフレームの作成
            fig, ax = plt.subplots()
            # step3 棒グラフの描画
            ax.barh(x, spp, tick_label=labels)
            ax.set_xlabel('average of absolute shap values')
            ax.set_ylabel('feature names')
            ax.set_title('bar plot')
            plt.show()   
        
        return 0
    
