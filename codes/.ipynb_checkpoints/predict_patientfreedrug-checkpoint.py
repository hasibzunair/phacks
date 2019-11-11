import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the psp data
# psp = pd.concat(pd.read_excel('PSP_Data_V3.xlsx', sheet_name=None, skiprows=1, index_col=False), ignore_index=False)
psp = pd.read_excel('Emi.xlsx', sheet_name="09-2019", skiprows=1, index_col=False)
print(psp.describe())

"""Preprocessing the data"""
# Define the training label and check for the class balance
target_label = psp['Patient Receiving Free Drug'].unique()
target_count = psp['Patient Receiving Free Drug'].value_counts()
print('Patient not receiving free drug: ', target_count[0])
print('Patient receiving free drug: ', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')
#Set tick colors:
ax = plt.gca()
ax.tick_params(axis='x', colors='red')
ax.tick_params(axis='y', colors='red')
target_count.plot(kind='bar', title='Total Number of target labels', color = 'b')
plt.grid()
plt.show()
#
# Put all the fraud class in a separate dataset.
psp['Patient Receiving Free Drug'][psp['Patient Receiving Free Drug'] == "Yes"] = 1
psp['Patient Receiving Free Drug'][psp['Patient Receiving Free Drug'] == "No"] = 0

yes_df = psp.loc[psp['Patient Receiving Free Drug'] == 1]
#Randomly select 37662 observations
no_df = psp.loc[psp['Patient Receiving Free Drug'] == 0].sample(n=target_count[1],random_state=42)

# Concatenate both dataframes again
normalized_df = pd.concat([yes_df, no_df])

#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))
sns.countplot('Patient Receiving Free Drug', data=normalized_df)
plt.title('Balanced Classes')
plt.show()

# Define the training label and check for the class balance
target_label_1 = normalized_df['Patient Receiving Free Drug'].unique()
target_count_1 = normalized_df['Patient Receiving Free Drug'].value_counts()
print('Patient not receiving free drug : ', target_count_1[0])
print('Patient receiving free drug : ', target_count_1[1])
print('Proportion:', round(target_count_1[0] / target_count_1[1], 2), ': 1')


normalized_df= normalized_df.drop(["Patient ID", "Day Enrollment Received", "Day Enrollment Completed", "On Drug Start Day", "Last On Drug Day", "State Change Day", "Re-Engagement Day", "Re-Engagement On Drug Start Day", "DR ID", "DR Speciality", "Status Description"], axis=1)
# normalized_df['Patient ID'] = normalized_df['Patient ID'].str.extract('(\d+)', expand=False)

normalized_df['Gender ID'][normalized_df['Gender ID']=="Gender_1"] = 1
normalized_df['Gender ID'][normalized_df['Gender ID']=="Gender_2"] = 2
normalized_df['Gender ID'][normalized_df['Gender ID']=="Unknown"] = float("Nan")


normalized_df['Age Range'][normalized_df['Age Range']== "0-9"] = 5
normalized_df['Age Range'][normalized_df['Age Range']== "10-19"] = 15
normalized_df['Age Range'][normalized_df['Age Range']== "20-29"] = 25
normalized_df['Age Range'][normalized_df['Age Range']== "30-39"] = 35
normalized_df['Age Range'][normalized_df['Age Range']== "40-49"] = 45
normalized_df['Age Range'][normalized_df['Age Range']== "50-59"] = 55
normalized_df['Age Range'][normalized_df['Age Range']== "60-69"] = 65
normalized_df['Age Range'][normalized_df['Age Range']== "70-79"] = 75
normalized_df['Age Range'][normalized_df['Age Range']== "80-89"] = 85
normalized_df['Age Range'][normalized_df['Age Range']== "90-99"] = 95
normalized_df['Age Range'][normalized_df['Age Range']== "100-109"] = 105
normalized_df['Age Range'][normalized_df['Age Range']== "110-119"] = 115
normalized_df['Age Range'][normalized_df['Age Range']=="Unknown"] = float("Nan")

normalized_df['Diagnosis ID'] = normalized_df['Diagnosis ID'].str.extract('(\d+)', expand=False)
normalized_df['Biologic Line of Therapy'] = normalized_df['Biologic Line of Therapy'].str.extract('(\d+)', expand=False)

normalized_df.Dosage = normalized_df.Dosage.str.extract('(\d+)', expand=False)
normalized_df.Frequency = normalized_df.Frequency.str.extract('(\d+)', expand=False)
normalized_df['Status Group'] = normalized_df['Status Group'].str.extract('(\d+)', expand=False)
normalized_df.Status = normalized_df.Status.str.extract('(\d+)', expand=False)
normalized_df['Case State'] = normalized_df['Case State'].str.extract('(\d+)', expand=False)
normalized_df['DR PROVINCE'] = normalized_df['DR PROVINCE'].str.extract('(\d+)', expand=False)


normalized_df['Payment Method #1'][normalized_df['Payment Method #1']=="Yes"] = 1
normalized_df['Payment Method #1'][normalized_df['Payment Method #1']=="No"] = 0
normalized_df['Payment Method #1'][normalized_df['Payment Method #1']=="Unknown"] = float("Nan")

normalized_df['Payment Method #2'][normalized_df['Payment Method #2']=="Yes"] = 1
normalized_df['Payment Method #2'][normalized_df['Payment Method #2']=="No"] = 0
normalized_df['Payment Method #2'][normalized_df['Payment Method #2']=="Unknown"] = float("Nan")

normalized_df['Payment Method #3'][normalized_df['Payment Method #3']=="Yes"] = 1
normalized_df['Payment Method #3'][normalized_df['Payment Method #3']=="No"] = 0
normalized_df['Payment Method #3'][normalized_df['Payment Method #3']=="Unknown"] = float("Nan")

normalized_df['Payment Method #4'][normalized_df['Payment Method #4']=="Yes"] = 1
normalized_df['Payment Method #4'][normalized_df['Payment Method #4']=="No"] = 0
normalized_df['Payment Method #4'][normalized_df['Payment Method #4']=="Unknown"] = float("Nan")

normalized_df['Payment Method #5'][normalized_df['Payment Method #5']=="Yes"] = 1
normalized_df['Payment Method #5'][normalized_df['Payment Method #5']=="No"] = 0
normalized_df['Payment Method #5'][normalized_df['Payment Method #5']=="Unknown"] = float("Nan")


# from xgboost import XGBClassifier
# # from sklearn.metrics import accuracy_score

normalized_df.fillna(method='ffill', inplace=True)

# Preparing Data

X = normalized_df.to_numpy()

from sklearn.decomposition import IncrementalPCA
import numpy as np
y_training = X[:,4].astype(np.int) #Training Labels
"""['GenderID', 'Age_Range', 'DiagnosisID', 'BLOT', 'Dosage',
       'Frequency', 'Status_Group', 'Status', 'Case_State', 'DR_ID',
       'DR_PROVINCE', 'DR_Speciality']"""
X_training = np.delete(X, [4], axis=1) #Training Features

# Splitting data for training in the ratio 70(training):10(validation):20(testing)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_training,y_training,test_size= 0.15, random_state=5)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=120,
                             random_state=0)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
target_names = ['class 0', 'class 1']
print(classification_report(y_test, y_pred, target_names=target_names))

from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
print(cm)
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4) #for label size
sns.heatmap(cm, annot=True)
plt.show()


