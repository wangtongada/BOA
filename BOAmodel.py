import pandas as pd 
from fim import fpgrowth,fim
import numpy as np
import math
from itertools import chain, combinations
import itertools
from numpy.random import random
from bisect import bisect_left
from random import sample
from scipy.stats.distributions import poisson, gamma, beta, bernoulli, binom
import time
import operator
from collections import Counter, defaultdict
from scipy.sparse import csc_matrix


class BOA(object):
    def __init__(self, binary_data,Y):
        self.itemMatrix = [[item for item in binary_data.columns if row[item] ==1] for i,row in binary_data.iterrows() ]  
        self.df = binary_data  
        self.Y = Y
        self.index = np.where(Y==1)[0]
        self.nindex = np.where(Y!=1)[0]
        self.attributeLevelNum = defaultdict(int) 
        self.attributeNames = []

        for i,name in enumerate(binary_data.columns):
          attribute = name.split('_')[0]
          self.attributeLevelNum[attribute] += 1
          self.attributeNames.append(attribute)
        self.attributeNames = list(set(self.attributeNames))
        

    def getPatternSpace(self):
        print 'Computing sizes for pattern space ...'
        start_time = time.time()
        """ compute the rule space from the levels in each attribute """
        patternSpace = np.zeros(self.maxlen+1)
        for k in xrange(1,self.maxlen+1,1):
            for subset in combinations(self.attributeNames,k):
                tmp = 1
                for i in subset:
                    tmp = tmp * self.attributeLevelNum[i]
                patternSpace[k] = patternSpace[k] + tmp
        print '\tTook %0.3fs to compute patternspace' % (time.time() - start_time)
        return patternSpace        

    def generate_rules(self,supp,maxlen,N):
        self.maxlen = maxlen
        print 'Generating rules...'
        start_time = time.time()
        rules= fpgrowth([self.itemMatrix[i] for i in self.index],supp = 5,zmin = 1,zmax = self.maxlen)
        start_time = time.time()
        print '\tTook %0.3fs to generate %d rules' % (time.time() - start_time, len(rules))
        print 'Selecting {} rules...'.format(N)
        itemInd = {}
        for i,name in enumerate(self.df.columns):
          itemInd[name] = i
        len_index = len(self.index)
        len_nindex = len(self.nindex)
        indices = np.array(list(itertools.chain.from_iterable([[itemInd[x] for x in rule[0]] for rule in rules])))
        len_rules = [len(rule[0]) for rule in rules]
        indptr =list(accumulate(len_rules))
        indptr.insert(0,0)
        indptr = np.array(indptr)
        data = np.ones(len(indices))
        ruleMatrix = csc_matrix((data,indices,indptr),shape = (len(self.df.columns),len(rules)))
        df = np.matrix(self.df.iloc[self.nindex])
        mat = df * ruleMatrix
        lenMatrix = np.matrix([len_rules for i in xrange(df.shape[0])])
        FP = np.array(np.sum(mat ==lenMatrix,axis = 0))[0]
        TP = np.array([rule[1][0] for rule in rules])
        TN = len_nindex - FP
        FN = len_index - TP
        p1 = TP.astype(float)/(TP+FP)
        p2 = FN.astype(float)/(FN+TN)
        pp = (TP+FP).astype(float)/(TP+FP+TN+FN)
        tpr = TP.astype(float)/(TP+FN)
        fpr = FP.astype(float)/(FP+TN)
        entropy = -pp*(p1*np.log(p1)+(1-p1)*np.log(1-p1))-(1-pp)*(p2*np.log(p2)+(1-p2)*np.log(1-p2))
        select = np.argsort(entropy)[::-1][-N:]
        self.rules = [rules[i][0] for i in select]
        print '\tTook %0.3fs to select %d rules' % (time.time() - start_time, min(len(rules),N))
        self.RMatrix = np.zeros([len(self.itemMatrix),len(self.rules)]) # for each observation, compare with all patterns to see if there's a match
        for i,row in enumerate(self.itemMatrix):
            self.RMatrix[i] = [set(rule).issubset(row) for rule in self.rules]
        self.patternSpace = self.getPatternSpace()
        self.precision = p1[select]


    def set_parameters(self, a1=100,b1=1,a2=1,b2=100,al=None,bl=None):
        # input al and bl are lists
        self.alpha_1 = a1
        self.beta_1 = b1
        self.alpha_2 = a2
        self.beta_2 = b2
        if al ==None or bl==None or len(al)!=self.maxlen or len(bl)!=self.maxlen:
            print 'No or wrong input for alpha_l and beta_l. The model will use default parameters!'
            self.C = [1.0/self.maxlen for i in xrange(self.maxlen)]
            self.C.insert(0,-1)
            self.alpha_l = [1 for i in xrange(self.maxlen+1)]
            self.beta_l= [self.patternSpace[i]/self.C[i] for i in xrange(self.maxlen+1)]
        else:
            self.alpha_l=al
            self.beta_l = bl

    def SA_patternbased(self, Niteration = 5000, Nchain = 3,print_message=True):
        print 'Searching for an optimal solution...'
        start_time = time.time()
        RMatrix = np.matrix(self.RMatrix)
        self.rules_len = [len(rule) for rule in self.rules]
        nRules = len(self.rules)
        maps = defaultdict(list)
        T0 = 100
        split = 0.85*Niteration
        q = 0.2 #indicates the level of randomization in annealing, can be user defined
        for chain in xrange(Nchain):
            # initialize with a random pattern set
            N = sample(xrange(1,8,1),1)[0]
            rules_curr = sample(xrange(nRules),N)
            rules_curr_norm = self.normalize(rules_curr)
            pt_curr = sum(self.compute_prob(rules_curr_norm)[1])
            maps[chain].append([-1,pt_curr,rules_curr,rules_curr_norm])

            for iter in xrange(Niteration):
                if iter<split:
                    rules_new = rules_curr[:]
                    rules_norm = rules_curr_norm[:]
                else:
                    p = np.array(xrange(1+len(maps[chain])))
                    p = np.array(list(accumulate(p)))
                    p = p/p[-1]
                    index = find_lt(p,random())
                    rules_new = maps[chain][index][2][:]
                    rules_norm = maps[chain][index][3][:]
                Yhat = (np.sum(self.RMatrix[:,rules_new],axis = 1)>0).astype(int)
                incorr = np.where(self.Y!=Yhat)[0]
                N = len(rules_new)
                if len(incorr)==0:
                    clean = True
                    move = ['clean']
                    # it means the HBOA correctly classified all points but there could be redundant patterns, so cleaning is needed
                else:
                    clean = False
                    ex = sample(incorr,1)[0]
                    t = random()
                    if self.Y[ex]==1 or N==1:
                        if t<1.0/2 or N==1:
                            move = ['add']       # action: add
                        else:
                            move = ['cut','add'] # action: replace
                    else:
                        if t<1.0/2:
                            move = ['cut']       # action: cut
                        else:
                            move = ['cut','add'] # action: replace
                if move[0]=='cut':
                    """ cut """
                    if random()<q:
                        candidate = list(set(np.where(self.RMatrix[ex,:]==1)[0]).intersection(rules_new))
                        if len(candidate)==0:
                            candidate = rules_new
                        cut_rule = sample(candidate,1)[0]
                    else:
                        p = []
                        all_sum = np.sum(self.RMatrix[:,rules_new],axis = 1)
                        for index,rule in enumerate(rules_new):
                            Yhat= ((all_sum - np.array(self.RMatrix[:,rule]))>0).astype(int)
                            TP,FP,TN,FN  = getConfusion(Yhat,self.Y)
                            p.append(log_betabin(TP,TP+FP,self.alpha_1,self.beta_1) + log_betabin(FN,FN+TN,self.alpha_2,self.beta_2))
                        p = [x - min(p) for x in p]
                        p = np.exp(p)
                        p = np.insert(p,0,0)
                        p = np.array(list(accumulate(p)))
                        if p[-1]==0:
                            index = sample(xrange(len(rules_new)),1)[0]
                        else:
                            p = p/p[-1]
                        index = find_lt(p,random())
                        cut_rule = rules_new[index]
                    rules_new.remove(cut_rule)
                    rules_norm = self.normalize(rules_new)
                    move.remove('cut')
                    
                if len(move)>0 and move[0]=='add':
                    """ add """
                    if random()<q:
                        add_rule = sample(xrange(nRules),1)[0]
                    else:
                        Yhat_neg_index = list(np.where(np.sum(self.RMatrix[:,rules_new],axis = 1)<1)[0])
                        mat = np.multiply(RMatrix[Yhat_neg_index,:].transpose(),self.Y[Yhat_neg_index]).transpose()
                        TP = np.array(np.sum(mat,axis = 0).tolist()[0])
                        FP = np.array((np.sum(self.RMatrix[Yhat_neg_index,:],axis = 0) - TP))
                        TN = np.sum(self.Y[Yhat_neg_index]==0)-FP
                        FN = sum(self.Y[Yhat_neg_index]) - TP
                        p = (TP.astype(float)/(TP+FP+1))
                        p[rules_new]=0
                        add_rule = sample(np.where(p==max(p))[0],1)[0]
                    if add_rule not in rules_new:
                        rules_new.append(add_rule)
                        rules_norm = self.normalize(rules_new)

                if len(move)>0 and move[0]=='clean':
                    remove = []
                    for i,rule in enumerate(rules_norm):
                        Yhat = (np.sum(self.RMatrix[:,[rule for j,rule in enumerate(rules_norm) if (j!=i and j not in remove)]],axis = 1)>0).astype(int)
                        TP,FP,TN,FN = getConfusion(Yhat,self.Y)
                        if TP+FP==0:
                            remove.append(i)
                    for x in remove:
                        rules_norm.remove(x)
                    return rules_norm
                        
                cfmatrix,prob =  self.compute_prob(rules_norm)
                T = T0**(1 - iter/Niteration)
                pt_new = sum(prob)
                alpha = np.exp(float(pt_new -pt_curr)/T)
                
                if pt_new > maps[chain][-1][1]:
                    maps[chain].append([iter,sum(prob),rules_new,rules_norm])
                    if print_message:
                        TP = cfmatrix[0]
                        FP = cfmatrix[1]
                        TN = cfmatrix[2]
                        FN = cfmatrix[3]
                        tpr = float(TP)/(TP + FN)
                        fpr = float(FP)/(FP + TN)
                        print '\n** chain = {}, max at iter = {} ** \nTP = {},FP = {}, TN = {}, FN = {}\n pt_new is {}, prior_ChsRules={}, likelihood_1 = {}, likelihood_2 = {}\n tpr = {}, fpr = {}'.format(chain, iter,cfmatrix[0],cfmatrix[1],cfmatrix[2],cfmatrix[3],sum(prob), prob[0], prob[1], prob[2],tpr,fpr)
                        self.print_rules(rules_new)

                if random() <= alpha:
                    rules_curr_norm,rules_curr,pt_curr = rules_norm[:],rules_new[:],pt_new

        pt_max = [maps[chain][-1][1] for chain in xrange(Nchain)]
        index = pt_max.index(max(pt_max))
        print '\tTook %0.3fs to generate an optimal rule set' % (time.time() - start_time)
        return maps[index][-1][3]

    def compute_prob(self,rules):
        Yhat = (np.sum(self.RMatrix[:,rules],axis = 1)>0).astype(int)
        TP,FP,TN,FN = getConfusion(Yhat,self.Y)
        Kn_count = list(np.bincount([self.rules_len[x] for x in rules], minlength = self.maxlen+1))
        prior_ChsRules= sum([log_betabin(Kn_count[i],self.patternSpace[i],self.alpha_l[i],self.beta_l[i]) for i in xrange(1,len(Kn_count),1)])            
        likelihood_1 =  log_betabin(TP,TP+FP,self.alpha_1,self.beta_1)
        likelihood_2 = log_betabin(FN,FN+TN,self.alpha_2,self.beta_2)
        post =  prior_ChsRules +  likelihood_1 + likelihood_2
        return [TP,FP,TN,FN],[prior_ChsRules,likelihood_1,likelihood_2]

    def normalize_add(self, rules_new, rule_index):
        rules = rules_new[:]
        for rule in rules_new:
            if set(self.rules[rule]).issubset(self.rules[rule_index]):
                return rules_new[:]
            if set(self.rules[rule_index]).issubset(self.rules[rule]):
                rules.remove(rule)
        rules.append(rule_index)
        return rules

    def normalize(self, rules_new):
        try:
            rules_len = [len(self.rules[index]) for index in rules_new]
            rules = [rules_new[i] for i in np.argsort(rules_len)[::-1][:len(rules_len)]]
            p1 = 0
            while p1<len(rules):
                for p2 in xrange(p1+1,len(rules),1):
                    if set(self.rules[rules[p2]]).issubset(set(self.rules[rules[p1]])):
                        rules.remove(rules[p1])
                        p1 -= 1
                        break
                p1 += 1
            return rules[:]
        except:
            return rules_new[:]


    def print_rules(self, rules_max):
        for rule_index in rules_max:
            print self.rules[rule_index]

def accumulate(iterable, func=operator.add):
    'Return running totals'
    # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
    # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
    it = iter(iterable)
    total = next(it)
    yield total
    for element in it:
        total = func(total, element)
        yield total

def find_lt(a, x):
    """ Find rightmost value less than x"""
    i = bisect_left(a, x)
    if i:
        return int(i-1)
    print 'in find_lt,{}'.format(a)
    raise ValueError


def log_gampoiss(k,alpha,beta):
    import math
    k = int(k)
    return math.lgamma(k+alpha)+alpha*np.log(beta)-math.lgamma(alpha)-math.lgamma(k+1)-(alpha+k)*np.log(1+beta)


def log_betabin(k,n,alpha,beta):
    import math
    try:
        Const =  math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    except:
        print 'alpha = {}, beta = {}'.format(alpha,beta)
    if isinstance(k,list) or isinstance(k,np.ndarray):
        if len(k)!=len(n):
            print 'length of k is %d and length of n is %d'%(len(k),len(n))
            raise ValueError
        lbeta = []
        for ki,ni in zip(k,n):
            lbeta.append(math.lgamma(ni+1)- math.lgamma(ki+1) - math.lgamma(ni-ki+1) + math.lgamma(ki+alpha) + math.lgamma(ni-ki+beta) - math.lgamma(ni+alpha+beta) + Const)
        return np.array(lbeta)
    else:
        return math.lgamma(n+1)- math.lgamma(k+1) - math.lgamma(n-k+1) + math.lgamma(k+alpha) + math.lgamma(n-k+beta) - math.lgamma(n+alpha+beta) + Const

def getConfusion(Yhat,Y):
    if len(Yhat)!=len(Y):
        raise NameError('Yhat has different length')
    TP = np.dot(np.array(Y),np.array(Yhat))
    FP = np.sum(Yhat) - TP
    TN = len(Y) - np.sum(Y)-FP
    FN = len(Yhat) - np.sum(Yhat) - TN
    return TP,FP,TN,FN

def predict(rules,df):
    Z = [[] for rule in rules]
    for i,rule in enumerate(rules):
        Z[i] = (np.sum(df[rule],axis=1)==len(rule)).astype(int)
    Yhat = (np.sum(Z,axis=0)>0).astype(int)
    return Yhat
