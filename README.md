####README

This code implements the Bayesian or-of-and algorithm as described in the BOA paper. We include the tictactoe dataset in the correct formatting to be used by this code.

This code requires the external frequent itemset mining package "PyFIM," available at http://www.borgelt.net/pyfim.html

It is specific to binary classification with binary features (although could 
easily be extended to multiclass).

##INPUT

The main code to be run by the user is example.py. This example.py uses input training data to generate ruled and then search for the optimal BOA using simmulated annealing. At the end of the file the true positive rate and false positive rate is computed to show the performance.
Notice that the input data has to be binary coded and the column names have to have the form 'attributename_attributevalue'. If your data is not binary coded, for example attributename #color has values {red, blue, yellow}, and an item "red" can be coded as "color_red"=1, "color_blue"=0, "color_yellow"=0. (The last two are optional. Usually including the absent/negative #items improve the predicting accuracy. Our example code does not use it for simplicity)

- tictactoe_X.txt : This is the file containing the X for tictactoe data, for which all features are binary. Each line is a data entry in which all of the features with value "1" are simply listed, with spaces delimiting different items. 'attributename_attributevalue' = 1 if attributename=attributevalue. For example, '1_X=1' means the 1st #position is X. 

- tictactoe_Y.txt : This file contains the Y data (labels), and contains a line corresponding to each line in tictactoe_X.txt. 

##OUTPUT

The function generate_rules generates patterns that satisfy the minimum support and maximum length and then select the Nrules rules that have the highest entropy. In function SA_patternbased, each local maximum is stored in maps and the best BOA is returned. Remember here the BOA contains only the index of selected rules from Nrules self.rules 
