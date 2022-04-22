from unittest import result
from flask import Flask, render_template, request
import pickle
import scipy.io
import flask
import joblib
import numpy as np
from scipy import misc


def create_app():
  app = Flask(__name__)
  model = pickle.load(open('model.pkl', 'rb'))

  @app.route('/')
  def index():
    return render_template('model2.html')

# 데이터 예측 처리
  @app.route('/predict', methods=['POST'])
  def home():
      data1 = request.form['type']
      data2 = request.form['country']
      data3 = request.form['release_year']
      data4 = request.form['title']

      arr = np.array ([[data1,data2,data3,data4]])
      pred = model.predict(arr)
      return render_template('hello.html', data = result)
  return app



if __name__ == "__main__":
    app = create_app()
    model = joblib.load('model/model.pkl')
    app.run(host="0.0.0.0", port=8000, debug=True)  # debug=True causes Restarting with stat