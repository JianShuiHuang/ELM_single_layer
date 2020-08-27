# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 16:58:04 2020

@author: ivis
"""

from ELM import *
from DataLoader import *
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torchvision import transforms
import time
from Time import *


##data loader
transformations = transforms.Compose([transforms.ToTensor()])
train_np = DataLoader('data/', 'train', transformations)
test_np = DataLoader('data/', 'test', transformations)


##ELM parameters
num_images_train = 28099
num_images_test = 7025
input_size = train_np[0][0].shape[0] * train_np[0][0].shape[1] * train_np[0][0].shape[2]
output_size = 5
hidden_size = 5000


##process data
"""
train_data = np.zeros((num_images_train, input_size))
train_label = np.zeros(num_images_train)
for i in range(num_images_train):
    if i % 100 == 0:
        print(i)
    train_data[i] = train_np[i][0].flatten()
    train_label[i] = 1 if train_np[i][1] >= 1 else 0

np.save("Retinopathy_train_data.npy", train_data)
np.save("Retinopathy_train_label.npy", train_label)


test_data = np.zeros((num_images_test, input_size))
test_label = np.zeros(num_images_test)
for i in range(num_images_test):
    if i % 100 == 0:
        print(i)
    test_data[i] = test_np[i][0].flatten()
    test_label[i] = 1 if test_np[i][1] >= 1 else 0

np.save("Retinopathy_test_data.npy", test_data)
np.save("Retinopathy_test_label.npy", test_label)
"""

train_data = np.load("Retinopathy_train_data.npy")
train_label = np.load("Retinopathy_train_label.npy")
test_data = np.load("Retinopathy_test_data.npy")
test_label = np.load("Retinopathy_test_label.npy")

print("data load complet...")

##create ELM model
model = ELM(input_size, output_size, hidden_size)
print("model complet...")

##train ELM model
start = time.time()

model.train(train_data, train_label.reshape(-1, 1))

print("Time: ", timeSince(start, 1 / 100))

##test ELM model
y_pred = model.predict(test_data)
y_pred = (y_pred > 0.5).astype(int)

##accuracy
print('Accuracy: ', accuracy_score(test_label, y_pred))

