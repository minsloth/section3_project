import os
from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    data10 = request.form['j']
    data11 = request.form['k']
    arr = np.array([[data1*365, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11]])
    pred = model.predict(np.float32(arr))
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)