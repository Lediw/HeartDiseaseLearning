import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import pickle

# Load the model from disk
with open('pretrained_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)

# Load the scaler
with open('scaler.sav', 'rb') as f:
    loaded_scaler = pickle.load(f)

while True:
    print('\n-------------Welcome to the heart disease helper!-------------\n')
    w = float(input('What is your weight in kilograms? \n'))
    h = float(input('What is your height in meters? \n'))
    bmi = w/h**2
    smoke = float(input('Do you smoke? (0 for no & 1 for yes )\n'))
    alcohol = float(input('Do you drink? (0 for no & 1 for yes )\n'))
    stroke = float(input('Have you had any stroke symptom? (0 for no & 1 for yes )\n'))
    walking = float(input('Do you have any difficulty walking? (0 for no & 1 for yes )\n'))
    sex = float(input('What is your sex? (0 for male & 1 for female)\n'))
    age = float(input('What is your age?\n'))
    diab = float(input('Are you diabetic? (0 for no & 1 for yes )\n'))
    act = float(input('Do you regularly have physical activities? (0 for no & 1 for yes )\n'))
    sleep = float(input('How many hours do you sleep per day?\n'))
    asm = float(input('Do you have asthma? (0 for no & 1 for yes )\n'))
    kidney = float(input('Do you have any kidney disease? (0 for no & 1 for yes )\n'))
    skin = float(input('Do you have skin cancer? (0 for no & 1 for yes )\n'))

    print('Hold on, calculating...\n')
    X = np.array([[bmi, smoke, alcohol, stroke, walking, sex, age, diab, act, sleep, asm, kidney, skin]]).reshape(1, -1)
    scaler = loaded_scaler
    X = scaler.transform(X)
    op = loaded_model.predict_proba(X)[:,1][0]

    print(f'Your probability of having a heart disease is {op*100:.4}%\n')

    print('Suggestions:\n')

    Suggestions = False

    if(smoke==1):
        X = np.array([[bmi, 0, alcohol, stroke, walking, sex, age, diab, act, sleep, asm, kidney, skin]]).reshape(1, -1)
        X = scaler.transform(X)
        p = loaded_model.predict_proba(X)[:,1][0] 
        if(p-op < 0):
            print(f'Stop smoking. This can change your probability of having a heart disease to {p*100:.4}%\n')
            Suggestions = True
	
    if(alcohol==1):
        X = np.array([bmi, smoke, 0, stroke, walking, sex, age, diab, act, sleep, asm, kidney, skin]).reshape(1, -1)
        X = scaler.transform(X)
        p = loaded_model.predict_proba(X)[:,1][0] 
        if(p-op < 0):
            print(f'Stop drinking. This can change your probability of having a heart disease to {p*100:.4}%\n')
            Suggestions = True

    if(act==0):
        X = np.array([[bmi, smoke, alcohol, stroke, walking, sex, age, diab, 1, sleep, asm, kidney, skin]]).reshape(1, -1)
        X = scaler.transform(X)
        p = loaded_model.predict_proba(X)[:,1][0] 
        if(p-op < 0):
            print(f'Do some exercise. This can change your probability of having a heart disease to {p*100:.4}%\n')
            Suggestions = True

    for i in range(12):
        X = np.array([[bmi, smoke, alcohol, stroke, walking, sex, age, diab, act, sleep+i, asm, kidney, skin]]).reshape(1, -1)
        X = scaler.transform(X)
        p = loaded_model.predict_proba(X)[:,1][0]
        if(op==0):
            continue
        if((op-p)/op > 0.05):
            print(f'Sleep {i} more hours per day. This can change your probability of having a heart disease to {p*100:.4}%\n')
            Suggestions = True
            break

    for i in range(20):
        X = np.array([[bmi-i/h**2, smoke, alcohol, stroke, walking, sex, age, diab, act, sleep, asm, kidney, skin]]).reshape(1, -1)
        X = scaler.transform(X)
        p = loaded_model.predict_proba(X)[:,1][0]
        if(op==0):
            continue
        if((op-p)/op > 0.05):
            print(f'Loss {i} kilograms of weight. This can change your probability of having a heart disease to {p*100:.4}%\n')
            Suggestions = True
            break

    for i in range(50):
        X = np.array([[bmi, smoke, alcohol, stroke, walking, sex, age+i, diab, act, sleep, asm, kidney, skin]]).reshape(1, -1)
        X = scaler.transform(X)
        p = loaded_model.predict_proba(X)[:,1][0]
        if(op==0):
            continue
        if(abs((op-p)/max(op, p)) > 0.1):
            print(f'In {i} years, your probability of having a heart disease will become {p*100:.4}%\n')
            Suggestions = True
            break
    if(not Suggestions):
        print(f'None')