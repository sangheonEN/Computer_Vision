# 1000 category classification, train 1200000, test 150000
# input image data : 224 * 224,  BGR Color
# output score : 1 * 1000

import sys
import cv2
import numpy as np

# 입력 영상 불러오기
# (CMD에서 1개의 IMAGE 파일 명으로 불러올 수 있게)

filename = 'space_shuttle.jpg'

if len(sys.argv) > 1:
    filename = sys.argv[1]

img = cv2.imread(filename)

if img is None:
    print("image load file")
    sys.exit()

# network 불러오기 google net (weight, bias 저장 파일 model에 저장 config 파일은 hyperparameters 저장)

# caffe
# model = 'googlenet/bvlc_googlenet.caffemodel'
# config = 'googlenet/deploy.prototxt'

# onnx
model = 'googlenet/googlenet-9.onnx'
config = ''

net = cv2.dnn.readNet(model, config)

if net.empty():
    print("network load fail")
    sys.exit()

# 클래스 이름 불러오기
classname = None
with open('googlenet/classification_classes_ILSVRC2012.txt', 'rt') as f:
    classname = f.read().rstrip('\n').split('\n')

# 추론
blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
net.setInput(blob)
prob = net.forward()

# 추론 결과 확인
out = prob.flatten()
classId = np.argmax(out)
confidence = out[classId]

text = f"{classname[classId]} : ({confidence * 100}%)"
cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imshow('img', img)
cv2.waitKey()

cv2.destroyAllWindows()