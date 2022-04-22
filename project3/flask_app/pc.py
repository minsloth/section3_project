# static에 고정적으로 들어가야할 사진 등이 들어간다
# templates 등에 html이 들어가는 구조이다.

from crypt import methods
from flask import Flask, render_template, request
import pickle
# from matplotlib.pyplot import title
import numpy as np


# 만들어온 모델은 다음과 같이 불러온다.
# 즉, 이미 저장되어 있어야한다.!

# model = pickle.load(open('model.pkl', 'rb'))

# 다음과 같이 Flask Module이 시작된다.
app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    request_method = request_method
    return render_template('hello.html',request_method=request_method)


if __name__ == "__main__":
    app.run(debug=True)


# 기본주소에서 일어나는 일
# @app.route('/')
# def main():
	# home.html을 띄워준다.
#    return render_template('model.html')




'''
    # predict 페이지에서는 POST라는 method가 사용될 것이다. 
@app.route('/predict', methods=['POST'])
def home():
    model = pickle.load(open('model.pkl', 'rb'))
    # 다음과 같이 Flask Module이 시작된다.
    app = Flask(__name__)

    data1 = request.form['type']
    data2 = request.form['counrty']
    data3 = request.form['release_year']
    data4 = request.form['title']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    # 보여줄 페이지 그리고 어떤 데이터를 넘길지에 대해서 확인한다....
    # 모델이 예측한 결과를 넘겨서 그 값에 따라 if문을 작성하게 한다.

    df = pd.DataFrame({'a':[type], 'b':[counrty], 'c':[release_year], 'd':[title]})

    y_pred = model.predict(df)[0]
    return render_template('result.html', result = np.round(y_pred))

    
'''
