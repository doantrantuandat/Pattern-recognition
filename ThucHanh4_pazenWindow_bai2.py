"""
Doan Tran Tuan Dat - 16035741 - DHKHMT12A

"""

import numpy as np
import itertools as it
from scipy.spatial import distance as dt
import random as rd
from prettytable import PrettyTable

# doc du lieu
classA = []
fileA = open("classA.txt")
for line in fileA:
    classA.append(line.strip().split())  
classB = []  
fileB = open("classB.txt")
for line in fileB:
    classB.append(line.strip().split())

# Ham chia du lieu train va test
def splitData(data, cut):
    """ tach du lieu """
    split = np.int(np.floor(len(data)/100*cut))
    return data[:split], data[split:]

a_train, a_test = splitData(classA, 70) # Train: 70% , test 30%
#chuyen ve array kieu du lieu float
a_train = np.array(a_train, dtype=float)
a_test = np.array(a_test, dtype=float)

print('So mau classA:', len(classA))
print('So mau tap train cua classA:', len(a_train))
print('So mau tap test cua classA:', len(a_test))

b_train, b_test = splitData(classB, 70)
b_train = np.array(b_train,dtype = float)
b_test = np.array(b_test,dtype = float)
print('So mauclassB:', len(classB))
print('So mau tap train cua classB:', len(b_train))
print('So mau tap test cua classB:', len(b_test))

def phuy(x, xi, h):
    a = x - xi
    at = a.T
    kc = np.exp((-1*at.dot(a)) / (2*h**2))
    if(kc < 0.5):
        return 1
    else:
        return 0

def prob(x, train, h):
    t = 0
    for i in range(len(train)):
        t = t + phuy(x, train[i, :], h)
    return t*1/(len(train) * h**2)

def pazen_gauss(x, trainA, trainB, h):
    nA = len(trainA)
    nB = len(trainB)
    pA = nA / (nA + nB)
    pB = 1 - pA
    if((prob(x, trainA, h)*pA) > (prob(x, trainB, h)*pB)):
        return 1
    else:
        return 0

# Khai bao h
h = 1
#h = 0.5
#h = 1.5

demA_test1 = 0
demB_test1 = 0
target_class_1 = []
for i in range(0, len(a_test)):
    if(pazen_gauss(a_test[i], a_train, b_train, h) == 1):
        target_class_1.append('A')
        demA_test1 = demA_test1 + 1
    else:
        target_class_1.append('B')
        demB_test1 = demB_test1 + 1

# in ra bang so sanh khi lay du lieu A de test
print("-------------------Table target test A---------------------")
t1 = PrettyTable(['x', 'y', 'target'])
for i in range(len(target_class_1)):
    t1.add_row([a_test[i][0], a_test[i][1],target_class_1[i]])
print(t1)


print('---------------------Test classA--------------------------')
print('Dung:', demA_test1)
print('Sai:', demB_test1)


# in ra bang so sanh khi lay du lieu B de test
demA_test2 = 0
demB_test2 = 0
target_class_2 = []
for i in range(0, len(b_test)):
    if(pazen_gauss(b_test[i], a_train, b_train, h) == 1):
        demA_test2 = demA_test2 + 1
        target_class_2.append('A')
    else:
        demB_test2 = demB_test2 + 1
        target_class_2.append('B')

print("-------------------Table target test B---------------------")
t2 = PrettyTable(['x', 'y', 'target'])
for i in range(len(target_class_2)):
    t2.add_row([b_test[i][0], b_test[i][1],target_class_2[i]])
print(t2)

print('---------------------Test classB--------------------------')
print('Dung: ', demB_test2)
print('Sai: ', demA_test2)
print('Phan tram loi =', ((demB_test1 + demA_test2) * 100) / (len(a_test) + len(b_test)), "%")

