import numpy as np
import pandas as pd
from pandas.io.parsers import read_csv
from BOAmodel import *
from collections import defaultdict


""" parameters """
# The following parameters are recommended to change depending on the size and complexity of the data
N = 2000      # number of rules to be used in SA_patternbased and also the output of generate_rules
Niteration = 500  # number of iterations in each chain
Nchain = 2         # number of chains in the simulated annealing search algorithm

supp = 5           # 5% is a generally good number. The higher this supp, the 'larger' a pattern is
maxlen = 3         # maxmum length of a pattern

# \rho = alpha/(alpha+beta). Make sure \rho is close to one when choosing alpha and beta. 
alpha_1 = 500       # alpha_+
beta_1 = 1          # beta_+
alpha_2 = 500         # alpha_-
beta_2 = 1       # beta_-

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
model.generate_rules(supp,maxlen,N)
model.set_parameters(alpha_1,beta_1,alpha_2,beta_2,None,None)
rules = model.SA_patternbased(Niteration,Nchain,print_message=True)

# test
Yhat = predict(rules,df.iloc[test_index])
TP,FP,TN,FN = getConfusion(Yhat,Y[test_index])
tpr = float(TP)/(TP+FN)
fpr = float(FP)/(FP+TN)
print 'TP = {}, FP = {}, TN = {}, FN = {} \n accuracy = {}, tpr = {}, fpr = {}'.format(TP,FP,TN,FN, float(TP+TN)/(TP+TN+FP+FN),tpr,fpr)
