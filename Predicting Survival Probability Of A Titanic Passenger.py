
# The program reads a CSV file containing all the passenger's information who were on board the Titanic on the day of the catastrophic event. The program creates a model using 'Random Forest Classifier' that reads the CSV file first and then could predict the surviving probability of six dummy passengers; that could be created based on passenger class, age, and sex. The output results are shown in form of a table along with the accuracy score of the model. The program aims to find any biases or regulations that may have been followed by the on-board staff that in the end had affected the fate of the passengers.

 
import fontstyle
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Creating test-passengers bellow in order to predict the survival rate :
    
''' # The Code Goes Like : Test-Passenger = {'Passenger Class': 1/2/3,
                                             'Age': any natural number,
                                             'Sex': 0/1 i.e Female/Male} # '''
                                             
#### Create Test-Passengers Bellow:
    
first_test_passenger = {'Passenger Class': 1,
                        'Age': 18,
                        'Sex': 0}
second_test_passenger = {'Passenger Class': 3,
                         'Age': 29,
                         'Sex': 1}
third_test_passenger = {'Passenger Class': 1,
                        'Age': 12,
                        'Sex': 0}
fourth_test_passenger = {'Passenger Class': 2,
                         'Age': 18,
                         'Sex': 0}
fifth_test_passenger = {'Passenger Class': 3,
                        'Age': 38,
                        'Sex': 1}
sixth_test_passenger = {'Passenger Class': 1,
                        'Age': 40,
                        'Sex': 1}

# Reading the CSV file:
    
data = pd.read_csv('Titanic Passenger List.csv')


# Replacing null values with the mean:
    
mean_age = round(data['Age'].mean())
data['Age'] = data['Age'].fillna(mean_age)


# Creating dummy columns:
    
dummies = pd.get_dummies(data['Sex'], drop_first= True)
data = pd.concat([data, dummies], axis= 'columns')


# Assigning X & Y variable for the model:
    
x = data[['Pclass', 'Age', 'male']].values
y = data['Survived'].values


# Creating a test-train split in order to get the accuracy score of the model:
    
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.2)


# Using grid-search cv to get the best parameter:
     
grid_search_cv = GridSearchCV(RandomForestClassifier(), {'n_estimators': np.arange(20,60,10)}, cv= 5)
grid_search_cv.fit(x,y)


# Creating a model:
    
model = RandomForestClassifier(n_estimators= grid_search_cv.best_params_['n_estimators'])
model.fit(x, y)


# Creating another model in order to get the accuracy score:
    
model_for_score = RandomForestClassifier(n_estimators= grid_search_cv.best_params_['n_estimators'])
model_for_score.fit(x_train, y_train)


# Getting the informations of the test_passengers and creating a list with the values:
    
first = [values for keys, values in first_test_passenger.items()]
second = [values for keys, values in second_test_passenger.items()]
third = [values for keys, values in third_test_passenger.items()]
fourth = [values for keys, values in fourth_test_passenger.items()]
fifth = [values for keys, values in fifth_test_passenger.items()]
sixth = [values for keys, values in sixth_test_passenger.items()]


# Creating a variable for sex of all the test-passengers: 

sex = [first[2], second[2], third[2], fourth[2], fifth[2], sixth[2]]


# Predicting & assigning the predicted probability of a test-passenger to a variable:  

prediction_probability_1 = model.predict_proba([first])
prediction_probability_2 = model.predict_proba([second])
prediction_probability_3 = model.predict_proba([third])
prediction_probability_4 = model.predict_proba([fourth])
prediction_probability_5 = model.predict_proba([fifth])
prediction_probability_6 = model.predict_proba([sixth])


# Predicting the final outcome -- survived/dead -- of a test-passenger & assigning the results to a variable:
     
predictions = [model.predict([first])[0], model.predict([second])[0], model.predict([third])[0], 
               model.predict([fourth])[0], model.predict([fifth])[0], model.predict([sixth])[0]]


# Creating a table to show the results:
    
myTable = PrettyTable()

#### Creating Columns For The Table:
    
columns = ['Passenger I.D', 'P-Class', 'Age', 'Sex', 'Probability Of Death (%)', 
           'Probability Of Survival (%)', 'Prediction']

#### Adding The Columns To The Table:
#... Adding The Sl.no/Passenger I.D ...#
myTable.add_column(columns[0], np.arange(1,7))
#... Adding The P-Class Of All The Test-Passengers ...#
myTable.add_column(columns[1], [first[0], second[0], third[0], fourth[0], fifth[0], sixth[0]])
#... Adding The Age Of All The Test-Passengers ...#
myTable.add_column(columns[2], [first[1], second[1], third[1], fourth[1], fifth[1], sixth[1]])
#... Converting The Gender Values From 0/1 to Female/Male For All The Test-Passengers & Adding It To The Table ...#
myTable.add_column(columns[3], ['Male' if i == 1 else 'Female' for i in sex])
#... Adding The Probability Of Death Of All The Test-Passenger ...# 
myTable.add_column(columns[4], [round(prediction_probability_1[0][0]*100,2), 
                                round(prediction_probability_2[0][0]*100,2),
                                round(prediction_probability_3[0][0]*100,2), 
                                round(prediction_probability_4[0][0]*100,2), 
                                round(prediction_probability_5[0][0]*100,2), 
                                round(prediction_probability_6[0][0]*100,2)])
#... Adding The Probability Of Survival Of All The Test-Passenger ...#                                
myTable.add_column(columns[5], [round(prediction_probability_1[0][1]*100,2), 
                                round(prediction_probability_2[0][1]*100,2), 
                                round(prediction_probability_3[0][1]*100,2), 
                                round(prediction_probability_4[0][1]*100,2), 
                                round(prediction_probability_5[0][1]*100,2), 
                                round(prediction_probability_6[0][1]*100,2)])
#... Adding The Final Prediction -- Dead/Survived -- Of All The Test Passengers ..#                            
myTable.add_column(columns[6], ['Survived' if i == 1 else 'Dead' for i in predictions])


# Creating a title:
    
title = fontstyle.apply('Predicting The Probability Of Survival Of A Test-Passenger In The Titanic Crash Using Random Forest Classifier:', 'green/bold/underline')


# Getting the n-estimator used in the model and the accuracy score:
     
n_estimator = grid_search_cv.best_params_['n_estimators']
model_accuracy = round(model_for_score.score(x_test,y_test)*100,2)


# Printing everything:
    
print(title)
print(fontstyle.apply('\n' + 'N-Estimatior Used: ', 'yellow') + str(n_estimator))
print(fontstyle.apply('Model Accuracy: ', 'yellow') + str(model_accuracy) + '%')
print('\n' + 'Table:')
print(myTable)


''' Created By Sourin Das '''
