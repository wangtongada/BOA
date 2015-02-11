import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from BOAmodel import *
from collections import defaultdict


""" parameters """
Nrules = 2000      # number of rules to be used in SA_patternbased and also the output of generate_rules
Niteration = 5000  # number of iterations in each chain
Nchain = 3         # number of chains in the simulated annealing search algorithm
supp = 5           # 5%
maxlen = 3         # maxmum length of a pattern
alpha1 = 500       # alpha_+
beta1 = 1          # beta_+
alpha2 = 1         # alpha_-
beta2 = 500        # beta_-

""" input file """
# notice that in the example, X is already binary coded. 
# Data has to be binary coded and the column name shd have the form: attributename_attributevalue
filepathX = 'tictactoe_X.txt' # input file X
filepathY = 'tictactoe_Y.txt' # input file Y
df = read_csv(filepathX,header=0,sep=" ")
Y = np.loadtxt(open(filepathY,"rb"),delimiter=" ")


lenY = len(Y)
train_index = sample(xrange(lenY),int(0.70*lenY))
test_index = [i for i in xrange(lenY) if i not in train_index]

model = BOA(df.iloc[train_index],Y[train_index])
model.generate_rules(supp,maxlen,Nrules)
model.set_parameters(alpha1,beta1,alpha2,beta2,None,None)
ruleset = model.SA_patternbased(Niteration,Nchain,print_message=True)

rules = [list(model.rules[i]) for i in ruleset]
Yhat = predict(rules,df.iloc[test_index])
TP,FP,TN,FN = getConfusion(Yhat,Y[test_index])
tpr = float(TP)/(TP+FN)
fpr = float(FP)/(FP+TN)
print 'tpr = {}, fpr = {}'.format(tpr,fpr)
