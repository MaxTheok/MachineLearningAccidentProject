"""
    Welcome to the UK accident Machine Learning report

    We used the following columns
    "Accident_Severity", "Number_of_Vehicles", "Number_of_Casualties", "Month", "Day_of_Week", "Hour",
    "Local_Authority_(District)", "1st_Road_Class", "1st_Road_Number", "Road_Type", "Speed_limit",
    "Junction_Control", "Pedestrian_Crossing-Human_Control", "Light_Conditions", "Weather_Conditions",
    "Road_Surface_Conditions","Special_Conditions_at_Site","Carriageway_Hazards", "Urban_or_Rural_Area",
    "Did_Police_Officer_Attend_Scene_of_Accident", "Year"

    DETAILS:
    "Accident_Severity": 1 Fatal, 2 Serious, 3 Slight
    "Day_of_Week": 1 Sunday ... 7 Saturday
    "Local_Authority_(District)": Distric numbers
    "1st_Road_Class": 1 Motorway, 2 A(M), 3 A, 4 B, 5 C, 6 Unclassified
    "Road_Type": Roundabout, 1 way stree, Dual Carriageway, Single carriageway, slip road, unknown, 1 way street slip road, data missing
    "Junction_Control": Not at junction or within 20 metres, Authorised person, Auto traffic signal, Stop sign, Give way or uncontrolled, Data missing or out of range
    "Pedestrian_Crossing-Human_Control": None within 50 metres , Control by school crossing patrol, Control by other authorised person, Data missing or out of range
    "Light_Conditions":Daylight, Darkness - lights lit, Darkness - lights unlit, Darkness - no lighting, Darkness - lighting unknown, Data missing or out of range
    "Weather_Conditions": Fine no high winds, Raining no high winds, Snowing no high winds, Fine + high winds, Raining + high winds, Snowing + high winds, Fog or mist, Other, Unknown, Data missing or out of range
    "Road_Surface_Conditions": Dry, Wet or damp, Wet or damp, Snow, Frost or ice, Flood over 3cm. deep, Oil or diesel,Mud,Data missing or out of range
    "Special_Conditions_at_Site":None, Auto traffic signal - out, Auto signal part defective, Road sign or marking defective or obscured, Roadworks, Road surface defective, Oil or diesel, Mud, Data missing or out of range
    "Carriageway_Hazards":None, Vehicle load on road, Other object on road, Previous accident, Dog on road, Other animal on road, Pedestrian in carriageway - not injured, Any animal in carriageway (except ridden horse), Data missing or out of range
    "Urban_or_Rural_Area": 1 Urban, 2 Rural, 3 None Allocated
    "Did_Police_Officer_Attend_Scene_of_Accident": Yes, No, accident was reported using a self completion  form (self rep only)


    We modifyed the Date and Time columns in excel as it was more effective than modifying them with our code. our for loops where very inefficient.
"""

import pandas as pd
import numpy as np
import datetime as dt

#Labels of the columns we will be using
ColumnsNames = [ "Accident_Severity","Number_of_Vehicles", "Number_of_Casualties", "Month", "Day_of_Week", "Hour", "Local_Authority_(District)", "1st_Road_Class", "1st_Road_Number", "Road_Type", "Speed_limit", "Junction_Control", "Pedestrian_Crossing-Human_Control", "Light_Conditions", "Weather_Conditions", "Road_Surface_Conditions","Special_Conditions_at_Site","Carriageway_Hazards", "Urban_or_Rural_Area", "Did_Police_Officer_Attend_Scene_of_Accident", "Year"]

#importing  the 3 csv files
data1 = pd.read_csv("accidents_2005_to_2007.csv", usecols = ColumnsNames)
data2 = pd.read_csv("accidents_2009_to_2011.csv", usecols = ColumnsNames)
data3 = pd.read_csv("accidents_2012_to_2014.csv", usecols = ColumnsNames)

#concatenating the data
dat = [data1, data2, data3]

data = pd.concat(dat)

#Transforming the data from value names to numbers for analysis.
data = data.reset_index()
data["Road_Type"] = data["Road_Type"].map({'Roundabout': 1, 'One way street': 2, 'Dual carriageway': 3, 'Single carriageway': 4, 'Slip road': 5 , 'Unknown': 0})
data["Junction_Control"] = data["Junction_Control"].map({'Automatic traffic signal': 1, 'Giveway or uncontrolled': 2, 'Stop Sign':3, 'Authorised person':4, })
data["Pedestrian_Crossing-Human_Control"] = data["Pedestrian_Crossing-Human_Control"].map({"None within 50 metres": 1 , "Control by school crossing patrol": 2, "Control by other authorised person": 3})
data["Light_Conditions"] = data["Light_Conditions"].map({'Daylight: Street light present': 1, 'Darkness: Street lights present and lit':2, 'Darkness: Street lighting unknown':3, 'Darkness: Street lights present but unlit':4, 'Darkeness: No street lighting':5})
data["Weather_Conditions"] = data["Weather_Conditions"].map({'Raining without high winds':1, 'Fine without high winds':2, 'Snowing without high winds':3, 'Fine with high winds':4, 'Raining with high winds':5, 'Fog or mist':6, 'Snowing with high winds' :7, 'Other':8, 'Unknown':0})
data["Road_Surface_Conditions"] = data["Road_Surface_Conditions"].map({"Dry":1, 'Wet/Damp':2, 'Frost/Ice':3, 'Snow':4, 'Flood (Over 3cm of water)':5})
data["Special_Conditions_at_Site"] = data["Special_Conditions_at_Site"].map({"None":1, 'Ol or diesel':2, 'Roadworks':3, 'Auto traffic signal partly defective':4, 'Road surface defective':5, 'Auto traffic singal out':6, 'Permanent sign or marking defective or obscured':7, 'Mud':8})
data["Carriageway_Hazards"] = data["Carriageway_Hazards"].map({'None':1, 'Other object in carriageway':2, 'Pedestrian in carriageway (not injured)':3, 'Dislodged vehicle load in carriageway':4,  'Involvement with previous accident':5, 'Any animal (except a ridden horse)':6})
data["Did_Police_Officer_Attend_Scene_of_Accident"] = data["Did_Police_Officer_Attend_Scene_of_Accident"].map({"Yes":1, "No":2, "accident was reported using a self completion  form (self rep only)": 0})

#replacing missing values with 0
data = data.fillna(0)

#looking at the correslation, out of curiosity
print(data.corr())

# seperating our target values from our data
target = data["Accident_Severity"]
data = data.drop(['index',"Accident_Severity"], axis = 1)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

#creating training and test splits
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)


#Creating 100 arrays for Cross Validation
X_folds = np.array_split(X_train, 100)
Y_folds = np.array_split(Y_train, 100)

from sklearn.neural_network import MLPClassifier

X_folds = np.array_split(X_train, 100)
Y_folds = np.array_split(Y_train, 100)

activations = ['identity', 'logistic', 'tanh', 'relu']
solvers =['lbfgs', 'sgd', 'adam']
hidden_layer_sizes = [(5, 2),(10, 2),(20, 8),(2,5),(10,20)]

i = 0
for a in activations:
    for s in solvers:
        print("*"*100)
        for hls in hidden_layer_sizes:
            clf = MLPClassifier( alpha=1e-5,hidden_layer_sizes=hls, random_state=1, solver=s)

            print("Neural Network: ", "{0:.4f}".format(clf.fit(X_folds[i], Y_folds[i]).score(X_test,Y_test)), " activation = ",  a, " solver = ", s, " hidden_layer_sizes = ", hls)

            i += 1


threshholds = [-100,-50,-10,-5,-1,0,1,5,10,50,100]

print("For the Nearest centroid method")
for t in threshholds:
    from sklearn.neighbors.nearest_centroid import NearestCentroid
    clf = NearestCentroid(metric='manhattan',shrink_threshold=None)
    #clf.fit(X_folds[i],Y_folds[i])
    clf.fit(X_train,Y_train)
    #i += 1
    print("Our accuracy is ", "{0:.4f}".format(clf.score(X_test,Y_test)), "for a threshhold of: ", t)


from sklearn.ensemble import RandomForestClassifier

i = 0
scoresForest = []
scoresSVC = []
params = []

# Number of trees in the forest
n_estimators = [5,10,20,100,200]
max_depth = [None,2,5,10,15,20]
bootstrap = [True, False]


for b in bootstrap:
    for md in max_depth:
        print("*"*100)
        for ne in n_estimators:
            clf = RandomForestClassifier(n_estimators=ne, max_depth=md, random_state=0, bootstrap=b)
            scoresForest.append(clf.fit(X_folds[i],Y_folds[i]).score(X_test,Y_test))
            print("The forest method wields an accuracy of: ", "{0:.4f}".format(clf.fit(X_folds[i],Y_folds[i]).score(X_test,Y_test)), "when max_depth = ",  md, "and bootstrap = ", b, "estimator = ", ne)
            i += 1


# Support vector classification, doesn't work with big samples
# My computer is possibly not powerful enough to run this.

'''degree = [2,5,10,15,20]
probability = [True, False]
kernel = ["linear", "poly", "rbf", "sigmoid", "precomputed"]
penalty = [1,5,10,15,20]
X_folds = np.array_split(X_train, 10000)
Y_folds = np.array_split(Y_train, 10000)
scoresSVC = []

i = 0
for d in degree:
    for p in probability:
        for k in kernel:
            print("*"*100)
            clf2 = SVC(gamma='auto', degree=d, probability=p, kernel = k, C=p)
            scoresSVC.append(clf2.fit(X_folds[i],Y_folds[i]).score(X_test,Y_test))
            print("Support Vector Classification accuracy: ", "{0:.4f}".format(clf2.fit(X_folds[i],Y_folds[i]).score(X_test,Y_test)), " degree = ",  d, " probability = ", p, " kernel =", k)
            i += 1  '''
