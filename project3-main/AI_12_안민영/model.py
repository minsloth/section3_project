# data set : https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset/code?select=cardio_train.csv


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#데이터 불러오기
df = pd.read_csv('./cardio_train.csv')

#필요없는 Feature 제거
df.drop('id',axis=1, inplace = True)


#훈련, 테스트, 검증 데이터
x_train,x_test = train_test_split(df, test_size=0.3) 
x_train,x_val = train_test_split(x_train, test_size = 0.3)

#타겟값 설정
target = 'cardio'

#훈련
y_train = x_train[target]
x_train = x_train.drop(target, axis = 1)
#테스트
y_test = x_test[target]
x_test = x_test.drop(target,axis =1)
#검증
y_val = x_val[target]
x_val = x_val.drop(target,axis=1)

#모델 학습(XGBoost)
model = XGBClassifier(
    n_estimators=1000,  # <= 1000 트리로 설정했지만, early stopping 에 따라 조절됩니다.
    max_depth=5,        # .
    learning_rate=0.2,
    #scale_pos_weight = 'ratio', # imbalance 데이터 일 경우 비율을 적용합니다.
    n_jobs=-1
)

eval_set = [(x_train, y_train), 
            (x_val, y_val)]

model.fit(x_train, y_train, 
          eval_set=eval_set,
          eval_metric='error', # #(wrong cases)/#(all cases)
          early_stopping_rounds = 50)


y_pred = [y_train.mode()[0]]*len(y_train)
Accuracy = accuracy_score(y_train, y_pred)
print('기준모델 :', Accuracy)


print('모델 검증 정확도 :', model.score(x_val, y_val))

#부호화
import pickle

with open('model.pkl','wb') as pickle_file:
    pickle.dump(model, pickle_file)