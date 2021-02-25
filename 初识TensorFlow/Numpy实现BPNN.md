```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def create_date():
    np.random.seed(1)
    m = 400  # 数量
    D = 2   # 维度
    N = int(m / 2)  # 每个标签的实例数
    X = np.zeros((m, D))
    y = np.zeros((m, 1), dtype='uint8')

    for j in range(2):
        ix = range(N*j, N*(j+1))
        # 返回3.12 均分为N段间隔的数据
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = 4 * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    X = X.T
    y = y.T
    return X, y

X, y = create_date()

# plt.scatter(X[0, :], X[1, :], c=y[0], s=40, cmap=plt.cm.Spectral)
# plt.show()

# 构建BP神经网络模型步骤 1.构建前向传 2.损失函数定义及计算 3.更新反向传播权重  4.循环更新

# 定义网络属性
def layer_sizes(X,y):
    n_x = X.shape[0] # 输入层维度
    n_h = 4 # 隐藏层节点数量
    n_y = y.shape[0] # 输出层维度

    return (n_x,n_h,n_y)


# 网络参数初始化
def initialize_parameters(n_x, n_h, n_y):

    w1 = np.random.randn(n_h,n_x) * 0.01 # w1为输入层至隐藏层的权重
    b1 = np.zeros((n_h,1)) # b1 为输入层至隐藏层的偏置变量
    w2 = np.random.randn(n_y,n_h) * 0.01 # b1为隐藏层至输出层的权重
    b2 = np.zeros((n_y,1)) # b2 为隐藏层至输出层的偏置变量

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))

    params = {
        'w1': w1,
        'b1': b1,
        'w2': w2,
        'b2': b2
    }

    return params



# 前向传播
def forward_propagation(x,params):
    w1 = params['w1']
    b1 = params['b1']
    w2 = params['w2']
    b2 = params['b2']
    z1 = np.dot(w1,x) + b1  # 隐藏层输入
    A1 = np.tanh(z1)  # 隐藏层激活函数处理

    z2 = np.dot(w2,A1) + b2  # 输出层输入
    A2 = np.tanh(z2)  # 输出层激活函数处理

    assert (A2.shape == (1, x.shape[1]))

    cache = {
        'z1': z1,
        'A1': A1,
        'z2': z2,
        'A2': A2
    }

    return A2, cache

# 计算损失函数
def compute_cost(A2,Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)
    cost = -1 / m * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost


# 根据损失值调整权值
def backforward_propagation(parameters,cache,x,y):
    m = x.shape[1]
    w1 = parameters['w1']
    w2 = parameters['w2']

    A1 = cache['A1']
    A2 = cache['A2']

    dz2 = A2 - y
    dw2 = 1/m * np.dot(dz2, A1.T)
    db2 = 1/m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(w2.T, dz2) * (1-np.power(A1, 2))
    dw1 = 1 / m * np.dot(dz1, x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    grads = {
        'dw1': dw1,
        'db1': db1,
        'dw2': dw2,
        'db2': db2
    }
    return grads


def update_params(params,grads,learning_rate = 1.2):
    w1 = params['w1']
    w2 = params['w2']
    b1 = params['b1']
    b2 = params['b2']

    dw1 = grads['dw1']
    dw2 = grads['dw2']
    db1 = grads['db1']
    db2 = grads['db2']

    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2

    params = {
        'w1': w1,
        'w2': w2,
        'b1': b1,
        'b2': b2
    }
    return params



def nn_model(X,y,n_h,num_iterations = 1000, print_cost = False):
    np.random.seed(3)
    n_x = layer_sizes(X, y)[0]
    n_y = layer_sizes(X, y)[2]
    parameters = initialize_parameters(n_x,n_h,n_y)

    for i in range(num_iterations):
        A2, cache = forward_propagation(X,parameters)
        if i % 100 == 0:
            cost = compute_cost(A2,y)
            print(cost)

        grads = backforward_propagation(parameters,cache,X,y)
        parameters = update_params(parameters, grads, learning_rate=1.2)

    return parameters


print(nn_model(X,y,4))

```
