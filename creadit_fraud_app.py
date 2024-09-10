from flask import Flask, flash, render_template, request, url_for, redirect, session
import bcrypt
from pymongo import MongoClient
import pickle
import pandas as pd
import numpy as np
import random
app = Flask(__name__)
app.secret_key = 'my_secret_key' 

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['mydatabase']
users_collection = db['users']

#Loading The Model
model_file = 'Credit_card_fraud_detection.pkl'
model = pickle.load(open(model_file,'rb'))


#Default Route
@app.route('/')
def index():
        return render_template('index.html')
#Prediction
@app.route('/prediction')
def predict():
    return render_template('prediction.html')

@app.route('/predict_fraud',methods=['POST'])
def predict_fraud():
    if request.method == 'POST':
        type = int(request.form.get('type'))
        amount = int(request.form.get('amount'))
        oldbalanceOrg = int(request.form.get('oldbalanceOrg'))
        newbalanceOrig = int(request.form.get('newbalanceOrig'))
    prediction = model.predict([[type,amount,oldbalanceOrg,newbalanceOrig]])


    #Return the prediction result
    return render_template('prediction.html',is_fraud = prediction[0])


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        
        # Check if username or email already exists
        if users_collection.find_one({'$or': [{'username': username}, {'email': email}]}):
            return 'Username or email already exists!'
        
        # Encrypt password
        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())
        
        # Insert new user into the database
        users_collection.insert_one({'username': username, 'email': email, 'password': hashed_password})
        return render_template('login.html')

    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        # Get form data
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        
        # Check if user exists and credentials are correct
        user = users_collection.find_one({'email': email})
        if user and bcrypt.checkpw(password, user['password']):
            session['email'] = email
            return redirect(url_for('home.html'))

        
    
    return render_template('login.html')
@app.route('/home',methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(debug=True)

