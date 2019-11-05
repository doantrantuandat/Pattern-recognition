"""
Doan Tran Tuan Dat - 16035741 - DHKHMT12A

"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math

# khoi tao du lieu
class_a = [1,2,3,3,4,4,6,6,8]
class_b = [4,6,7,7,8,9,9,10,12]

# im loi
def gx(xt):
    loia = 0
    loib = 0
    for i in class_a:
        if i > xt:
            loia = loia + 1  
    for i in class_b:
        if i <= xt:
            loib = loib + 1
    return loia, loib

loi_a = []
loi_b = []
bien = 0
# gop du lieu 2 mang va set lai du lieu khong bi trung
dataset = np.concatenate((class_a, class_b))
dataset = set(dataset)

for xt in dataset:
    loia, loib = gx(xt)
    loi_a.append(loia)
    loi_b.append(loib)


# tim min loi va ra ket qua bien phan lop
min = 10000000000
for xt in dataset:
    loia, loib = gx(xt)
    if min > loia + loib:
        min = loia + loib
        bien = xt 
print("bien can tim la: ", bien)


