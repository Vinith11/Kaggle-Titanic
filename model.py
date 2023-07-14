import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)

df=pd.read_csv('train.csv')
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age
def clean(df):
    
    df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)
    cols=['SibSp','Parch','Fare']
    for col in cols:
        df[col].fillna(df[col].median(),inplace=True)

    df.Embarked.fillna("U",inplace=True)
    embark=pd.get_dummies(df['Embarked'],drop_first=True)
    sex=pd.get_dummies(df['Sex'],drop_first=True)
    df=pd.concat([df,sex,embark],axis=1)
    
    return df

df=clean(df)

df=df.drop(['Ticket','Cabin','Name','PassengerId','Sex','Fare','Embarked'],axis=1)

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X=df.drop("Survived",axis=1)
y=df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

lr=LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)

prediction=lr.predict(X_test)


dat=[[1,25,0,0,0,0,1,0]]
print(lr.predict(dat))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        pclass = int(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        male = int(request.form['male'])
        q = int(request.form['q'])
        s = int(request.form['s'])

        # Prepare the input features as a numpy array
        input_features = np.array([[pclass, age, sibsp, parch, male, q, s]])
        prediction = lr.predict(input_features)

        # Determine the predicted result
        if prediction[0] == 1:
            result = 'Survived'
        else:
            result = 'Did not survive'

        return render_template('index.html', prediction_result=result)

if __name__ == '__main__':
    app.run(debug=True)
