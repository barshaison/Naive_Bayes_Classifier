import pandas as pd
import os
import re #for splitting
import PreProcessing as pp
import Classifier as cl
from Tkinter import *
from tkFileDialog import askdirectory
from tkMessageBox import *

pathToTrain = ""
pathToTest = ""
pathToStructure = ""


# Gui:
master = Tk()
master.wm_title("Naive Bayes Classifier")
master.configure(background='white')
master.geometry("400x300")
Label(master,text="",background='white').grid(row=0)
Label(master, text="Directory Path:",background='white').grid(row=1)
Label(master, text="Discretization Bins:",background='white').grid(row=2)

e1 = Entry(master)
e1.configure(bd=4,width=35)
e2 = Entry(master)
e2.configure(bd=4)

fileMissing = 0

def choose_directory():
    global dir_path
    dir_path = askdirectory()
    e1.insert(10,dir_path)
    global pathToTrain
    global pathToTest
    global pathToStructure
    pathToTest = dir_path + "/test.csv"
    pathToTrain = dir_path + "/train.csv"
    pathToStructure = dir_path + "/Structure.txt"
    global fileMissing
    fileMissing = 0
    errorString = "The following files are missing:\n"
    try:
        f = open(pathToTest)
    except IOError as e:
        fileMissing = 1
        errorString += "test.csv\n"
    try:
        f = open(pathToTrain)
    except IOError as e:
        fileMissing = 1
        errorString += "train.csv\n"
    try:
        f = open(pathToStructure)
    except IOError as e:
        fileMissing = 1
        errorString += "Structure.txt\n"
    if fileMissing == 1:
        build_Button.config(state='disabled')
        showinfo("Naive Bayes Classifier",errorString)
    else:
        build_Button.config(state='normal')

def setNumOfBins():
    global numOfIntervals
    numOfIntervals = int(e2.get())

def build_handler():

    try:
        #setNumOfBins()
        global numOfIntervals
        toCheck = e2.get()
        if toCheck == "":
            showinfo("Naive Bayes Classifier", "Please insert an integer for the Discretization bins attribute")
            return
        numOfIntervals = int(toCheck)
    except:
        showinfo("Naive Bayes Classifier", "Discretization bins must be an integer")
        return
    if numOfIntervals < 2:
        showinfo("Naive Bayes Classifier", "Discretization bins must be at least 2")
        return

    if os.stat(pathToStructure).st_size == 0:
        showinfo("Naive Bayes Classifier", "The file Structure.txt is empty. Please load valid files")
        return
    structure_file = open(pathToStructure, "r")
    try:
        dfTrain = pd.read_csv(pathToTrain)
    except Exception as e:
        if e.__str__() == "No columns to parse from file":
            showinfo("Naive Bayes Classifier", "The file train.csv is empty. Please load valid files")
        else:
            showinfo("Naive Bayes Classifier", "The file train.csv has errors. Please load valid files")
    totalNumOfRecords_train = dfTrain.shape[0]  # num of records
    if numOfIntervals > totalNumOfRecords_train:
        showinfo("Naive Bayes Classifier", "Discretization bins must not be grater than the number of train set records")
        return
    global dfTrainFinal
    dfTrainFinal = pp.preProcess(structure_file, dfTrain, numOfIntervals)
    structure_file = open(pathToStructure, "r")
    attribute_values_dict = pp.set_attribute_values_dict(structure_file)

    cl.prepareModel(dfTrainFinal, pathToStructure, numOfIntervals,attribute_values_dict)
    classify_Button.config(state='normal')
    showinfo("Naive Bayes Classifier","Building classifier using train-set is done!")

def classify_handler():
    try:
        dfTest = pd.read_csv(pathToTest)
    except Exception as e:
        if e.__str__() == "No columns to parse from file":
            showinfo("Naive Bayes Classifier", "The file test.csv is empty. Please load valid files")
        else:
            showinfo("Naive Bayes Classifier", "The file test.csv has errors. Please load valid files")

    structure_file = open(pathToStructure, "r")
    dfTestFinal = pp.preProcess_test(structure_file, dfTest)
    dfTestFinal.__delitem__('class')  # remove redundant class column
    try:
        cl.classify(dfTestFinal, dfTrainFinal, dir_path)
        classify_Button.config(state='disabled')
        showinfo("Naive Bayes Classifier", "Classification process of test-set is done!")
    except Exception as e:
        showinfo("Naive Bayes Classifier", "Incompatibale discretization bins value! Please enter smaller value ")



build_Button = Button(master, text="Build", command=build_handler,width=25)
classify_Button = Button(master, text="Classify", command=classify_handler,width=25)
e1.grid(row=1, column=1)
e2.grid(row=2, column=1)
Label(master,text="",background='white').grid(row=3)
build_Button.grid(row=4,column=1)
build_Button.config(state='disabled')

Label(master,text="",background='white').grid(row=5)
classify_Button.grid(row=6,column=1)
classify_Button.config(state='disabled')
browse_Button = Button(master, text="Browse", command=choose_directory)
browse_Button.grid(row=1,column=2)

mainloop()
















