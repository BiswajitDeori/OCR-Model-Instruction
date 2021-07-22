
import  os
import  cv2 as cv
import matplotlib.pyplot as plt
import  numpy as np
import tensorflow.keras.optimizers

from  sklearn.model_selection import train_test_split
path= 'math_sign/train'

classes=[]

for i in os.listdir(path):
    classes.append(i)


x_t=[]
x_lebal=[]

for i in classes:
    for images in os.listdir('math_sign/train/'+i):
        img=cv.imread('math_sign/train/'+i+'/'+images)
        x_lebal.append(int(i))
        img=cv.resize(img,(32,32))
        x_t.append(img)


x_t=np.array(x_t)
x_lebal=np.array(x_lebal)


x_train,x_test,y_train,y_test=train_test_split(x_t,x_lebal,test_size=0.2)
x_train,x_valid,y_train,y_valid=train_test_split(x_t,x_lebal,test_size=0.2)


def preProccesing(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.equalizeHist(img)
    img=img/255
    return img

# img=preProccesing(x_train[dott])
#
# img=cv.resize(img,(300,300))
# cv.imshow('img',img)
# cv.waitKey(0)


x_train=np.array(list(map(preProccesing,x_train)))
x_test=np.array(list(map(preProccesing,x_test)))
x_valid=np.array(list(map(preProccesing,x_valid)))


x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_valid=x_valid.reshape(x_valid.shape[0],x_valid.shape[1],x_valid.shape[2],1)
from  tensorflow import keras
from tensorflow.python.keras.utils import generic_utils
from  tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train,17)
y_test=to_categorical(y_test,17)
y_valid=to_categorical(y_valid,17)



from  tensorflow.keras.preprocessing.image import  ImageDataGenerator

datGen=ImageDataGenerator(zoom_range=0.2,shear_range=0.1,rotation_range=15)

from  tensorflow.keras.models import Sequential


################################# fro pre train model load########################
model=tensorflow.keras.models.load_model('simple_model1')
# ###############################################################
#
#
# ######################################
pre=[]

import numpy as np

img = cv.imread("filefile.jpg")

# grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

# binary
# ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 35, 180)
# cv.imshow('threshold', thresh)

# dilation
kernel = np.ones((1, 1), np.uint8)
img_dilation = cv.dilate(thresh, kernel, iterations=1)
# cv.imshow('dilated', img_dilation)

# find contours
# cv2.findCountours() function changed from OpenCV3 to OpenCV4: now it have only two parameters instead of 3
cv2MajorVersion = cv.__version__.split(".")[0]
# check for contours on thresh
if int(cv2MajorVersion) >= 4:
    ctrs, hier = cv.findContours(img_dilation.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
else:
    im2, ctrs, hier = cv.findContours(img_dilation.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# sort contours
sorted_ctrs = sorted(ctrs, key=lambda ctr: cv.boundingRect(ctr)[0])

crops=[]

for i, ctr in enumerate(sorted_ctrs):
    # Get bounding box
    x, y, w, h = cv.boundingRect(ctr)

    # Getting ROI
    roi = img[y:y + h, x:x + w]

    # show ROI
    # cv2.imshow('segment no:'+str(i),roi)
    cv.rectangle(img, (x-12, y-12), (x + w+12, y + h+12), (0, 255, 0), 1)
    crop= img[y - 12:y + 12 + h, x - 12:x + w + 12]
    img1 = cv.resize(crop, (32, 32))
    img1 = preProccesing(img1)
    img1 = img1.reshape(1, 32, 32, 1)
    predection = model.predict(img1)
    pre1 = np.argmax(predection)
    pre.append(pre1)


    # if you want to save the letters without green bounding box, comment the line above
    if w > 5:
        cv.imwrite('C:\\Users\\PC\\Desktop\\output\\{}.png'.format(i), roi)


# print(len(crops))
cv.imwrite('static/after.png',img)
cv.imshow('marked areas', img)
cv.waitKey(0)




####################################





from tensorflow.keras.models import Sequential
from  tensorflow.keras.layers import Activation,Dense,Conv2D,MaxPool2D,Flatten,Dropout
from  tensorflow.keras.losses import categorical_crossentropy


############### train model #############################
# model=Sequential()
#
# model.add((Conv2D(filters=64,kernel_size=(3,3),input_shape=(32,32,1),activation='relu')))
#
# model.add((Conv2D(filters=64,kernel_size=(3,3),activation='relu')))
# model.add(MaxPool2D(pool_size=(2,2),strides=1))
# model.add((Conv2D(filters=32,kernel_size=(3,3),activation='relu')))
# model.add(MaxPool2D(pool_size=(2,2),strides=1))
#
# model.add(Dropout(.5))
#
# model.add(Flatten())
# model.add(Dense(units=500,activation='relu'))
# model.add(Dropout(.5))
#
# model.add(Dense(units=17,activation='softmax'))
#
# model.summary()
#
# model.compile(optimizer=tensorflow.keras.optimizers.Adam(lr=1e-5),loss='categorical_crossentropy',metrics=['accuracy'])
#
# model.summary()
#
# model.fit_generator(datGen.flow(x_train,y_train,batch_size=10),epochs=70,verbose=2,validation_data=(x_valid,y_valid))



######################### end #######################################################
####### for save train model ############
# model.save('simple_model1')
# print("model save ")
# #################################







# pre=[]

#################################
# img1=cv.imread('no_check.png')
# gray=cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
# # gray=cv.GaussianBlur(gray,(5,5),.5)
# mser=cv.MSER_create()
# clone=img1.copy()
# region,_ = mser.detectRegions(gray)
# hulls=[cv.convexHull(p.reshape(-1,1,2)) for p in region]
# for i,j in enumerate(hulls):
#     x,y,z,w=cv.boundingRect(j)
#     crops = clone[y-12:y+12 + w, x-12:x + z+12]
#     clone=cv.rectangle(clone,(x,y),(x+z,y+z),(0,211,0),thickness=1)
#     img1 = cv.resize(crops, (32, 32))
#     img1 = preProccesing(img1)
#     img1 = img1.reshape(1, 32, 32, 1)
#     predection = model.predict(img1)
#     pre1 = np.argmax(predection)
#     if pre1 not in pre:
#         pre.append(pre1)
#
#
# cv.imshow('img',clone);
# cv.waitKey(0);
# cv.destroyAllWindows()









#
#
# predection = model.predict(img1)
# pre1 = np.argmax(predection)
# print(pre1)
#
#
data=[]

#################################

for i in pre:
    if i == 10:
        data.append('x')
    elif i == 11:
        data.append('/')
    elif i == 12:
        data.append('=')
    elif i==13:
        data.append('-')
    elif i == 14:
        data.append('+')
    elif i == 15:
        data.append('(')
    elif i == 16:
        data.append(')')
    else:
        data.append(i)


print(data)













#








