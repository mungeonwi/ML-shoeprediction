
import torch
import torch.nn as nn
import torch.optim as optim
import flask
import imageio
from flask import Flask, request, render_template
from torchvision import models
import torchvision.transforms as transforms

import numpy as np
   
app = Flask(__name__)
   
# index 페이지 라우팅
@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('shoeweb.html')
   
# 이미지 업로드에 대한 예측값 반환
@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method == 'POST':
   
        # 업로드 파일 처리 분기
        file = request.files['image']
        if not file: return render_template('shoeweb.html', label="No Files")
   
        # 이미지 픽셀 정보 읽기
        # 알파 채널 값 제거 후 1차원 Reshape
        img = imageio.imread(file)
        img = img[:, :, :3]
        img = img.reshape(1, -1)
   
        # 입력 받은 이미지 예측
        prediction = model.predict(img)
   
        # 예측 값을 1차원 배열로부터 확인 가능한 문자열로 변환
        label = str(np.squeeze(prediction))
   
        # 숫자가 10일 경우 0으로 처리
        if label == '10': label = '0'
   
        # 결과 리턴
        return render_template('shoeweb.html', label=label)
   
# 미리 학습시켜서 만들어둔 모델 로드
if __name__ == '__main__':
    model = torch.load('./model.pth')
    app.run(host='0.0.0.0', port=8000, debug=True)