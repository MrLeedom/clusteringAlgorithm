# -*- coding:utf-8 -*-

'''
作者：leedom
这段代码功能：将训练样本在坐标轴中显示出来
'''
from matplotlib import pyplot as plt
import numpy as np

#定义四个点的坐标
a1 = np.array([1,1])
a2 = np.array([1,2])
b1 = np.array([3,3])
b2 = np.array([3,4])
c = np.array([2,1])

#四个点的坐标分别赋值给X,Y
X1,Y1 = a1
X2,Y2 = a2
X3,Y3 = b1
X4,Y4 = b2
X5,Y5 = c
plt.figure()
plt.title('show data')
plt.scatter(X1,Y1,color='blue',label='a1')
plt.scatter(X2,Y2,color='blue',label='a2')
plt.scatter(X3,Y3,color='red',label='a3')
plt.scatter(X4,Y4,color='red',label='a4')
plt.scatter(X5,Y5,color='yellow',label='c')
plt.legend(loc='best')

plt.annotate(r'a1(1,1)',xy=(X1,Y1),xycoords='data',xytext=(+10,+30),textcoords='offset points',arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
plt.annotate(r'a2(1,2)',xy=(X2,Y2),xycoords='data',xytext=(+10,+30),textcoords='offset points',arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
plt.annotate(r'b1(3,3)',xy=(X3,Y3),xycoords='data',xytext=(+10,+30),textcoords='offset points',arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
plt.annotate(r'b2(3,4)',xy=(X4,Y4),xycoords='data',xytext=(+10,+30),textcoords='offset points',arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
plt.annotate(r'c(2,1)',xy=(X5,Y5),xycoords='data',xytext=(+10,+30),textcoords='offset points',arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

# plt.show()

'''
这段代码功能:计算某测试点到样本点的距离
'''
import math

def Euclidean(vec1,vec2):
    npvec1,npvec2 = np.array(vec1),np.array(vec2)
    return math.sqrt(((npvec1-npvec2)**2).sum())

def show_distance(test_point,c):
    line_point = np.array([test_point,c])
    x = (line_point.T)[0]
    y = (line_point.T)[1]
    #计算距离
    distance = round(Euclidean(line_point,c),2)
    #计算中点的位置用来存放距离的文字表示
    mid_x,mid_y = (test_point+c)/2
    plt.annotate(r'distance=%s'% str(distance),xy=(mid_x,mid_y),xycoords='data',xytext=(+10,+0),textcoords='offset points',fontsize=10,arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
    return plt.plot(x,y,linestyle='--',color='black',lw=1)

show_distance(a1,c)
show_distance(a2,c)
show_distance(b1,c)
show_distance(b2,c)
plt.show()