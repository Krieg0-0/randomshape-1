# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:45:50 2020

@author: 薛
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import graham_scan
pi = np.pi
cos = math.cos
sin = math.sin


# 旋转矩阵
def rotation_matrix(rotation):
    R = rotation
    matrix = [[cos(R), -sin(R)],
              [sin(R), cos(R)]]
    return np.array(matrix)


# 凸多边形面积公式
def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


# %%参数生成
def parameter(n, space, s):
    # n生成图形的数量. space画布的区域边长. s每个图形的面积.

    # 位置  n生成图形的数量. space画布的区域.
    X = np.random.rand(n) * space
    Y = np.random.rand(n) * space
    X_int = X.astype(int)
    Y_int = Y.astype(int)
    # X横坐标. Y纵坐标.

    # 找出凸包对应的点  key点的下标. X点的横坐标. Y点的纵坐标.
    r = s**0.5
    Xhull = np.asmatrix([[x-r+1, x-r, x+r+1, x+r] for x in X]).ravel().tolist()[0]
    Yhull = np.asmatrix([[y-r+1, y+r, y+r+1, y-r] for y in Y]).ravel().tolist()[0]
    key = np.linspace(1, len(Xhull), len(Xhull))
    point_array = [graham_scan.Point(key[i], Xhull[i], Yhull[i]) \
                   for i in range(len(Xhull))]
    hull = graham_scan.graham_scan(point_array)
    hull_X = [hull[i].get_x() for i in range(len(hull))]
    hull_Y = [hull[i].get_y() for i in range(len(hull))]
    hull_S = PolyArea(hull_X, hull_Y)
    # hull_X凸包的横坐标. hull_Y凸包的纵坐标. hull_S凸包的面积.

    # 矩形， 圆形， 三角形， 椭圆， 梯形， 平行四边形
    shape = ['rectangle', 'circle', 'triangle',
             'ellipse', 'trapezoid', 'parallelogram']

    # 形状与旋转
    shapes_code = [np.random.randint(0, len(shape)) for x in range(n)]
    shapes = [shape[i] for i in shapes_code]
    rotation = []
    for i in shapes:
        if (i == 'rectangle') | (i == 'ellipse') | (i == 'parallelogram'):
            rotation.append(np.random.rand() * pi)
        elif i == 'circle':
            rotation.append(0)
        elif i == 'triangle':
            rotation.append(np.random.rand() * pi * 2 / 3)
        elif i == 'trapezoid':
            rotation.append(np.random.rand() * pi * 2)
    # shapes形状. rotation旋转.

    # 长宽比
    p = []
    for i in shapes:
        if (i == 'rectangle') | (i == 'ellipse'):
            p.append(1 - np.random.rand() / 3)
        elif (i == 'trapezoid') | (i == 'parallelogram'):
            p.append(1 / 3)
        elif i == 'circle':
            p.append(1)
        elif i == 'triangle':
            p.append(2 / 3**0.5)
    # p长宽比.

    # 颜色
    color_name = ['blue', 'orange', 'green', 'red',
                  'purple', 'brown', 'pink', 'gray',
                  'olive', 'cyan']
    color_index = [np.random.randint(0, len(color_name)) for x in range(n)]
    color = [color_name[i] for i in color_index]
    # color图形的颜色.

    return X_int, Y_int, shapes, rotation, p, color, s, space, hull_S


# %%判断
def judge_circle(s, p):
    # s面积. p长宽比.
    rr = int((s / pi)**(0.5))
    r1, r2 = (rr, rr)
    return r1, r2


def judge_ellipse(s, p):
    # s面积. p长宽比.
    r1 = int((s/p/pi)**(0.5))
    r2 = int((s*p/pi)**(0.5))
    return r1, r2


def judge_rectangle(s, p):
    # s面积. p长宽比.
    r1 = int((s/p)**0.5/2)
    r2 = int((s*p)**0.5/2)
    return r1, r2


def judge_parallelogram(s, p):
    # s面积. p长宽比.
    r1 = int((s/2)**0.5*3/2)
    r2 = int((s/2)**0.5/2)
    return r1, r2


def judge_trapezoid(s, p):
    # s面积. p长宽比.
    r1 = int((s/2)**0.5*3/2)
    r2 = int((s/2)**0.5/2)
    return r1, r2


def judge_triangle(s, p):
    # x,y位置. s面积. p长宽比.
    r1 = int((s/(3**0.5))**0.5)
    r2 = int(r1*(3**0.5)/2)
    return r1, r2


def judge(X, Y, shapes, rotation, p, color, s, space, distance):
    # x,y位置. shapes图形形状. rotation旋转. p长宽比. s每个图形的面积. space画布的区域边长.

    # 用于判断生成参数的函数
    JSHAPE = dict(rectangle=judge_rectangle, circle=judge_circle,
                  triangle=judge_triangle, ellipse=judge_ellipse,
                  trapezoid=judge_trapezoid, parallelogram=judge_parallelogram)

    # 判别重叠画布矩阵的定义
    Matrix = [[0 for x in range(space)] for y in range(space)]
    Matrix = np.array(Matrix)
    # flag_1 判断是否超出边界, flag-2 判断是否重叠
    flag_1 = True
    flag_2 = True
    # 对于每一个图形的循环
    for i in range(len(X)):
        r1, r2 = JSHAPE[shapes[i]](s, p[i])
        # 判断是否超过画布边界
        rr = int((r1*r1+r2*r2)**(0.5))
        if (X[i]+rr >= space) | (Y[i]+rr >= space) | (X[i]-rr <= 0) | (Y[i]-rr <= 0):
            flag_1 = False
        else:
            # 判断是否重叠
            # 判断重叠图形矩阵生成
            ojm = np.array([[ii, jj] for ii in range(-r1, r1) for jj in range(-r2, r2)])
            ojm = np.dot(ojm, rotation_matrix(rotation[i]))
            ojm = ojm.astype(int)
            ojm = [[X[i], Y[i]] for ii in range(-r1, r1) for jj in range(-r2, r2)] + ojm
            ojm = np.unique(ojm, axis=0)
            # 判断是否重叠主体部分
            flag = True
            for zz in range(len(ojm)):
                if Matrix[ojm[zz][0], ojm[zz][1]] == 1:
                    for m in range(zz):
                        Matrix[ojm[m][0], ojm[m][1]] = Matrix[ojm[m][0], ojm[m][1]] - 1 
                    flag = flag_2 & False
                    break
                else:
                    Matrix[ojm[zz][0], ojm[zz][1]] = Matrix[ojm[zz][0], ojm[zz][1]] + 1
                    flag = flag_2 & True
            flag_2 = flag
    # 判断图形间的距离是否合适
    flag_3 = True
    for i in range(len(X)):
        X2 = [(X[i] - ii)**2 for ii in X]
        Y2 = [(Y[i] - ii)**2 for ii in Y]
        TotalDistance2 = X2 + Y2
        maxDistance = max(TotalDistance2)**0.5
    if maxDistance < (distance + s**0.5 * 2):
        flag_3 = False
    return flag_1 & flag_2 & flag_3


# %%画图
def parameter_circle(x, y, s, p, rotation):
    # x,y为生成图形位置. s为生成图形面积. p为长宽比. rotation为旋转.
    M = 64
    T = M * 2  # T为拟合的边框数
    xx = np.arange(T-1)
    yy = np.arange(T-1)
    angle = np.arange(0, 2*pi, pi/M)
    rr = int((s / pi)**(0.5))
    for i in range(T-1):
        xx[i] = x + rr*cos(angle[i])
        yy[i] = y + rr*sin(angle[i])
    return xx, yy


def parameter_ellipse(x, y, s, p, rotation):
    # x,y为生成图形位置. s为生成图形面积. p为长宽比. rotation为旋转.
    R = -rotation  # 椭圆参数方程的锅！
    M = 64
    T = M * 2
    xx = np.arange(T-1)
    yy = np.arange(T-1)
    angle = np.arange(0, 2*pi, pi/M)
    r1 = int((s/p/pi)**(0.5))
    r2 = int((s*p/pi)**(0.5))
    for i in range(T-1):
        t = angle[i]
        xx[i] = r1*cos(t)*cos(R)-r2*sin(t)*sin(R) + x
        yy[i] = r1*cos(t)*sin(R)+r2*sin(t)*cos(R) + y
    return xx, yy


def parameter_rectangle(x, y, s, p, rotation):
    # x,y为生成图形位置. s为生成图形面积. p为长宽比. rotation为旋转.
    R = rotation
    xx = np.arange(4)
    yy = np.arange(4)
    r1 = int((s/p)**(0.5)/2)
    r2 = int((s*p)**(0.5)/2)
    tx = [r1, -r1, -r1, r1]
    ty = [r2, r2, -r2, -r2]
    for i in range(4):
        xx[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[0] + x
        yy[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[1] + y
    return xx, yy


def parameter_parallelogram(x, y, s, p, rotation):
    # x,y为生成图形位置. s为生成图形面积. p为长宽比. rotation为旋转.
    R = rotation
    xx = np.arange(4)
    yy = np.arange(4)
    r1 = int((s/2)**0.5*3/2)
    r2 = int((s/2)**0.5/2)
    r3 = int((s/2)**0.5)
    tx = [r1-r3, -r1, -r1+r3, r1]
    ty = [r2, r2, -r2, -r2]
    for i in range(4):
        xx[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[0] + x
        yy[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[1] + y
    return xx, yy


def parameter_trapezoid(x, y, s, p, rotation):
    # x,y为生成图形位置. s为生成图形面积. p为长宽比. rotation为旋转.
    R = rotation
    xx = np.arange(4)
    yy = np.arange(4)
    r1 = int((s/2)**0.5*3/2)
    r2 = int((s/2)**0.5/2)
    r3 = r2
    tx = [r3, -r3, -r1, r1]
    ty = [r2, r2, -r2, -r2]
    for i in range(4):
        xx[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[0] + x
        yy[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[1] + y
    return xx, yy


def parameter_triangle(x, y, s, p, rotation):
    # x,y为生成图形位置. s为生成图形面积. p为长宽比. rotation为旋转.
    R = rotation
    xx = np.arange(3)
    yy = np.arange(3)
    r1 = int((s/(3**0.5))**0.5)
    r2 = int(r1*(3**0.5)/2)
    tx = [0, r1, -r1]
    ty = [r2, -r2, -r2]
    for i in range(3):
        xx[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[0] + x
        yy[i] = np.dot(np.array([tx[i], ty[i]]).T, rotation_matrix(R))[1] + y
    return xx, yy



def draw(X, Y, shapes, rotation, p, color, s, space, index):
    # x,y位置. shapes图形形状. rotation旋转. p长宽比. color颜色. s每个图形的面积. space画布的区域边长. index图片命名的索引

    # 生成画图所需参数的函数
    SHAPE = dict(rectangle=parameter_rectangle, circle=parameter_circle,
                 triangle=parameter_triangle, ellipse=parameter_ellipse,
                 trapezoid=parameter_trapezoid,
                 parallelogram=parameter_parallelogram)
    SHAPE_CHOICES = list(SHAPE.values())

    # matplotlib所能识别的颜色
    color_choice = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                    'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                    'tab:olive', 'tab:cyan']
    color_name = ['blue', 'orange', 'green', 'red',
                  'purple', 'brown', 'pink', 'gray',
                  'olive', 'cyan']
    color_dict = dict(zip(color_name, color_choice))
    # 画图的准备
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for i in range(len(X)):
        parameter_shape = SHAPE[shapes[i]]
        xx, yy = parameter_shape(X[i], Y[i], s, p[i], rotation[i])
        ax.fill(xx, yy, color_dict[color[i]])
    ax.set_xlim(0, space)
    ax.set_ylim(0, space)
    plt.axis('off')
    plt.savefig(f'num{len(X)}index{index}.png', dpi=500, bbox_inches='tight')
    plt.close('all')


## %%生成并保存
#def generator(N, n, space, s):
#    # N生成图片的数量. n生成图形的数量. space画布的区域边长. S每个图形的面积.
#    for i in range(N):
#        X, Y, shapes, rotation, p, color, s, space, hull_S = parameter(n, space, s)
#        if judge(X, Y, shapes, rotation, p, color, s, space, 30):
#            draw(X, Y, shapes, rotation, p, color, s, space, i)
#
#
#if __name__ == '__main__':
#    generator(100, 30, 720, 250)
#

# %%
if __name__ == '__main__':
    N, n, space, s = 10, 30, 720, 250
    flag = 1
    while flag <= N:
        X, Y, shapes, rotation, p, color, s, space, hull_S = parameter(n, space, s)
        if judge(X, Y, shapes, rotation, p, color, s, space, 30):
            draw(X, Y, shapes, rotation, p, color, s, space, flag)
            with open(f"num{n}index{flag}.csv", "w", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['X', 'Y', 'shapes', 'rotation', 'p', 'color'])
                for i in range(len(X)):
                    writer.writerow([X[i], Y[i], shapes[i], rotation[i], p[i], color[i]])
                writer.writerow(['num_of_pot', n, 'space', space, 'S_of_pot', s, 'S_of_hull', hull_S])
                csvfile.close()
            flag = flag +1
