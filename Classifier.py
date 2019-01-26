import pandas as pd
import re #for splitting
import PreProcessing as pp

#Classifier:
m = 2

classDict = {}# ['yes': 23, 'no':45] the n parameter
classes_probabilities = {}  # {c_1: P(c_1) ....}
attribute_domain_size_dict = {}  # ['default':2, ... , class:2]
n_c_dict = {}

def set_Nc_dict(dfTrainFinal,attribute_values_dict):
    global n_c_dict
    for att in dfTrainFinal:
        n_c_dict[att] = {}
        for c in classDict:
            n_c_dict[att][c]={}
            for val in attribute_values_dict[att]:
                n_c_dict[att][c][val] = count_records_with_both_values(att, val, 'class', c,dfTrainFinal)



def prepareModel(dfTrainFinal,pathToStructure,numOfIntervals,attribute_values_dict):

    totalNumOfRecords_train = dfTrainFinal.shape[0]  # num of records
    global classDict

    classDict = dfTrainFinal['class'].value_counts().to_dict()  # ['yes': 23, 'no':45] the n parameter
    class_probabilities_dict = {}
    # create the p(c_1), p(c_2) , ... probabilities
    for key in classDict:
        currProb = ((classDict[key]) / float(totalNumOfRecords_train))
        class_probabilities_dict[key] = currProb

    # create the M (domain size) of each attribute
    structure_file = open(pathToStructure, "r")

    for line in structure_file:
        splitedLine = re.split('\s', line)
        if splitedLine[2] != "NUMERIC":
            attribute_domain_size_dict[splitedLine[1]] = splitedLine[2].split(',').__len__()
        else:
            attribute_domain_size_dict[splitedLine[1]] = numOfIntervals

    numOfClasses = attribute_domain_size_dict['class']

    # zero step probabilities
    for c in classDict:
        classes_probabilities[c] = (classDict[c]) / float(totalNumOfRecords_train)

    set_Nc_dict(dfTrainFinal, attribute_values_dict)



# calculate n_c
def count_records_with_both_values(att1, val1, att2, val2,dfTrainFinal):
    tempDF = dfTrainFinal.loc[(dfTrainFinal[att1] == val1) & (dfTrainFinal[att2] == val2)]
    return tempDF.shape[0]


def classify(dfTestFinal,dfTrainFinal, dir_path):

    totalNumOfRecords_test = dfTestFinal.shape[0]  # num of records
    text_file = open(dir_path + "//output.txt", "w")
    for i in range(0, totalNumOfRecords_test):  # iterate over rows
        currRow = {}  # {att1:val1,....att_n=val_n}
        for col in dfTestFinal:  # iterate over cols
            currRow[col] = dfTestFinal[col][i]
        firstStepDict = {}  # {att : dict with probabilities of first step}
        for att in currRow:  # iterate over att of the curr row
            currAtt = {}  # {p(currAtt = v | class = v) : p}
            for c in classDict:  # iterate over classes
                n_c = n_c_dict[att][c][currRow[att]]#count_records_with_both_values(att, currRow[att], 'class', c,dfTrainFinal)
                M = attribute_domain_size_dict[att]
                p = (1 / float(M))
                n = classDict[c]
                p_currAtt_currClass = ((n_c + m * p) / float(n + m))
                currAtt[c] = p_currAtt_currClass
            firstStepDict[att] = currAtt
        second_Step_Probabilities_dict = {}  # dict of {c_i : P(X|c_i)}

        for c in classDict:  # iterate over classes
            currP = 1
            for att in firstStepDict:
                currP *= firstStepDict[att][c]
            second_Step_Probabilities_dict[c] = currP
        third_step_probabilities = {}  # {c_1: P(c_1)*P(X | c_1) ...}
        for c in classDict:
            third_step_probabilities[c] = (classes_probabilities[c]) * (second_Step_Probabilities_dict[c])
        currMaxClass = max(third_step_probabilities,
                           key=lambda i: third_step_probabilities[i])  # the chosen class (decision)

        #text_file.write(str(i+1) + " " + currMaxClass + "\n")
        text_file.write(currMaxClass + "\n")







