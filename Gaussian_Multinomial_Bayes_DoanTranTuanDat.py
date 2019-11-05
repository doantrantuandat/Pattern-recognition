"""
Doan Tran Tuan Dat - 16035741 - DHKHMT12A

"""
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
import numpy as np
import math as mt
from prettytable import PrettyTable
# load du lieu
iris = datasets.load_iris()
#print(iris)
X = iris['data']
X = X[:, :2]
y = iris['target']
#print(y)
# x0 : lop 0
# x1 : lop 1
# x2 : lop 2
"""
Phan lop Gaussian Naive Bayes
"""
x0 = []
x1 = []
x2 = []
for i in range(0, 50):
    x0.append(X[i])
for i in range(50, 100):
    x1.append(X[i])
for i in range(100, 150):
    x2.append(X[i])

x0 = np.array(x0)
x1 = np.array(x1)
x2 = np.array(x2)

#tinh mean cho tung class
mean_x1_x0 = np.mean(x0[:, 0])
mean_x2_x0 = np.mean(x0[:, 1])
x0_mean = [mean_x1_x0, mean_x2_x0]

mean_x1_x1 = np.mean(x1[:, 0])
mean_x2_x1 = np.mean(x1[:, 1])
x1_mean = [mean_x1_x1, mean_x2_x1]

mean_x1_x2 = np.mean(x2[:, 0])
mean_x2_x2 = np.mean(x2[:, 1])
x2_mean = [mean_x1_x2, mean_x2_x2]

# ham tinh phuong sai
def tinhPhuongSai(x_mean, dataset, index):  
    phuongsai = 0
    for i in dataset:
        s = (i[index] - x_mean[index]) ** 2
        phuongsai += s
    return phuongsai / (len(dataset) - 1)

# ham tinh xac suat gauss
def px_gauss(x_mean, dataset, index, phuongsai):
    list_px = []
    for i in range(len(dataset)):
        w = 1/(mt.sqrt(2 * mt.pi * phuongsai))

        tu = -(dataset[i][index] - x_mean[index]) ** 2

        mau = 2 * phuongsai

        px = w * np.exp(tu/mau)
        list_px.append(px)
    return list_px


phuongsai_0_0 = tinhPhuongSai(x0_mean, x0, 0)
phuongsai_0_1 = tinhPhuongSai(x0_mean, x0, 1)
a = px_gauss(x0_mean, X, 0, phuongsai_0_0)
b = px_gauss(x0_mean, X, 1, phuongsai_0_1)


phuongsai_1_0 = tinhPhuongSai(x1_mean, x1, 0)
phuongsai_1_1 = tinhPhuongSai(x1_mean, x1, 1)
c = px_gauss(x1_mean, X, 0, phuongsai_1_0)
d = px_gauss(x1_mean, X, 1, phuongsai_1_1)


phuongsai_2_0 = tinhPhuongSai(x2_mean, x2, 0)
phuongsai_2_1 = tinhPhuongSai(x2_mean, x2, 1)
e = px_gauss(x2_mean, X, 0, phuongsai_2_0)
f = px_gauss(x2_mean, X, 1, phuongsai_2_1)

list_xacsuat_x0 = []
list_xacsuat_x1 = []
list_xacsuat_x2 = []

for i in range(len(a)):
    list_xacsuat_x0.append(a[i] * b[i])

for i in range(len(c)):
    list_xacsuat_x1.append(c[i] * d[i])

for i in range(len(e)):
    list_xacsuat_x2.append(e[i] * f[i])

# ham tinh max xac suat
def max_px_total(data1, data2, data3):
    max_list = []
    for i in range(len(data1)):
        a = data1[i]
        b = data2[i]
        c = data3[i]
        tam = max(a,b,c)
        max_list.append(tam)
    return max_list


px_total = max_px_total(list_xacsuat_x0, list_xacsuat_x1, list_xacsuat_x2)

# ham the hien lop cua du lieu sau khi phan lop gauss
def set_target(dataset, data1, data2, data3):
    y_after = []
    for i in range(len(dataset)):
        if dataset[i] == data1[i]:
            y_after.append(0)
        elif dataset[i] == data2[i]:
            y_after.append(1)
        elif dataset[i] == data3[i]:
            y_after.append(2)
    return y_after


y_gauss = set_target(px_total, list_xacsuat_x0,
                          list_xacsuat_x1, list_xacsuat_x2)

def dem_loi_Gaussian():
    dem = 0
    for i in range(len(y)):
        if y[i] != y_gauss[i]:
            dem +=1
    return dem
print("so loi Gaussian Naive Bayes: ",dem_loi_Gaussian())
print("phan tram loi Gaussian Naive Bayes = ", (dem_loi_Gaussian()*100)/len(X), "%")

"""
Plot du lieu truoc va sau khi phan lop Gaussian
"""
# plot du lieu truoc khi phan lop
plt.figure(1, figsize=(8, 6))
plt.subplot(211)
plt.title("3 class before classify - Gaussian")
x_min_before, x_max_before = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min_before, y_max_before = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min_before, x_max_before)
plt.ylim(y_min_before, y_max_before)
plt.xticks(())
plt.yticks(())

# plot du lieu sau phan lop
plt.subplot(212)
plt.title("3 class after classify - Gaussian")

x_min_after, x_max_after = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min_after, y_max_after = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.scatter(X[:, 0], X[:, 1], c=y_gauss, edgecolor='k')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min_after, x_max_after)
plt.ylim(y_min_after, y_max_after)
plt.xticks(())
plt.yticks(())
plt.show()

"""
Phan lop Multinomial Naive Bayes

"""


def dem(data, train, index):
    dem = 0
    for i in range(len(train)):
        if (train[i][index] == data):
            dem = dem + 1
    return dem

def phanlop_multinomial(data, data0, data1, data2):
    
    y_after = []

    for i in range(len(data)):
        p_0_0 = (dem(data[i][0], data0, 0) / len(data0)) 
        p_0_1 = (dem(data[i][0], data1, 0) / len(data1)) 
        p_0_2 = (dem(data[i][0], data2, 0) / len(data2)) 

        p_1_0 = (dem(data[i][1], data0, 1) / len(data0)) 
        p_1_1 = (dem(data[i][1], data1, 1) / len(data1)) 
        p_1_2 = (dem(data[i][1], data2, 1) / len(data2)) 

        p_0_total = p_0_0 * p_1_0
        p_1_total = p_0_1 * p_1_1
        p_2_total = p_0_2 * p_1_2

        p_max = max(p_0_total, p_1_total, p_2_total)

        if (p_max == p_0_total):
            y_after.append(0)
        elif p_max == p_1_total:
            y_after.append(1)
        elif p_max == p_2_total:
            y_after.append(2)
            
    return y_after

y_multinomial= phanlop_multinomial(X, x0, x1, x2) # target sau khi phan lop multinomial

def dem_loi_Multinomial():
    dem = 0
    for i in range(len(y_multinomial)):
        if y_multinomial[i] != y[i]:
            dem = dem + 1
    return dem
print("so loi Multinomial = ",dem_loi_Multinomial())
print("phan tram loi Multinomial = ", (dem_loi_Multinomial()/len(X))*100, "%")


"""
Plot du lieu truoc va sau khi phan lop Multinomial
"""
# plot truoc khi phan lop
plt.figure(2, figsize=(8, 6))
plt.subplot(211)
plt.title("3 class before classify - Multinomial")
x_min_before, x_max_before = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min_before, y_max_before = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(x_min_before, x_max_before)
plt.ylim(y_min_before, y_max_before)
plt.xticks(())
plt.yticks(())

# plot du lieu sau phan lop
plt.subplot(212)
plt.title("3 class after classify - Multinomial")

x_min_after, x_max_after = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min_after, y_max_after = X[:, 1].min() - .5, X[:, 1].max() + .5

plt.scatter(X[:, 0], X[:, 1], c=y_multinomial, edgecolor='k')

plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min_after, x_max_after)
plt.ylim(y_min_after, y_max_after)
plt.xticks(())
plt.yticks(())
plt.show()

"""
Bien doi du lieu tu so thanh chu
in du lieu table so sanh 
"""
def biendoi_data(data):
    list = []
    for i in range(len(data)):
        if data[i] == 0:
            list.append('setosa')
        elif data[i] == 1:
            list.append('versicolor')
        elif data[i] == 2:
            list.append('virginica')
    return list

y_biendoi = biendoi_data(y)
y_gauss_biendoi = biendoi_data(y_gauss)
y_mul_biendoi = biendoi_data(y_multinomial)
t = PrettyTable(['width', 'length', 'target tien nghiem', 'Gaussian Naive Bayes','Multinomial Naive Bayes'])
for i in range(len(X)):
    t.add_row([X[i][0], X[i][1], y_biendoi[i], y_gauss_biendoi[i],y_mul_biendoi[i]])
print(t)




