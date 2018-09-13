
# coding: utf-8

# # 一元线性回归

# 基本形式
# $$\tag{1}
# f(\mathbf{x}) = w_1 x_1 + w_2 x_2 + ... + w_d x_d + b
# $$
# 向量形式
# $$\tag{2}
# f(\mathbf{x}) = \mathbf{w}^\mathrm{T}\mathbf{x}+b
# $$
# 其中$\mathbf{w} = (w_1; w_2; ...; w_d)$.
# 
# 针对一元线性回归，我们有
# $$\tag{3}
# f(x) = w x + b$$

# ## 1. 读取数据

# In[1]:


import numpy as np


# In[2]:


import pandas as pd

# 读取数据
data = pd.read_csv('data/kaggle_house_price_prediction/kaggle_hourse_price_train.csv')

# 丢弃有缺失值的特征（列）
data.dropna(axis = 1, inplace = True)

# 只保留整数的特征
data = data[[col for col in data.dtypes.index if data.dtypes[col] == 'int64']]


# 本题的要求是，针对LotArea, BsmtUnfSF, GarageArea三个特征，使用训练集训练三个一元线性回归模型，对比其在测试集上的MAE和RMSE，并且要绘制曲线。

# 所以我们只保留这三列与标记这一列，总共四列的数据即可

# In[3]:


features = ['LotArea', 'BsmtUnfSF', 'GarageArea']
target = 'SalePrice'
data = data[features + [target]]


# ## 2. 打乱数据顺序

# In[4]:


from sklearn.utils import shuffle


# In[5]:


data_shuffled = shuffle(data, random_state = 32) # 这个32不要改变


# In[6]:


data_shuffled.head()


# ## 3. 取前70%的数据为训练集，后30%为测试集

# In[7]:


num_of_samples = data_shuffled.shape[0]
split_line = int(num_of_samples * 0.7)
train_data = data.iloc[:split_line]
test_data = data.iloc[split_line:]


# In[8]:


train_data.shape


# In[9]:


test_data.shape


# ## 4. 编写模型

# In[10]:





# 我们以类(class)的形式编写这个模型，python中的类很简单，只不过这个类里面需要调用两个函数，一个是get_w，这个是计算模型w的函数，另一个是get_b，计算模型b的函数，需要大家来完成。

# 本实验要求使用最小二乘法求解一元线性回归模型  
# 求解$w$和$b$使均方误差$E_{(w,b)} = \sum^m_{i=1}(y_i - wx_i - b)^2$最小化的过程，称为线性回归模型的最小二乘“参数估计”(parameter estimation)。我们可将$E_{(w,b)}$分别对$w$和$b$求导，得到
# $$\tag{4}
# \frac{\partial E_{(w,b)}}{\partial w} = 2(w \sum^m_{i=1} x^2_i - \sum^m_{i=1} (y_i - b) x_i),
# $$
# 
# $$\tag{5}
# \frac{\partial E_{(w,b)}}{\partial b} = 2(mb - \sum^m_{i=1}(y_i - w x_i))
# $$
# 
# 然后令式(4)和式(5)为0，可得到$w$和$b$的闭式解(closed-form solution)
# $$\tag{6}
# w = \frac{\sum^m_{i=1} y_i(x_i - \bar{x})}{\sum^m_{i=1}x^2_i - \frac{1}{m}(\sum^m_{i=1}x_i)^2}
# $$
# 
# $$\tag{7}
# b = \frac{1}{m}\sum^m_{i=1}(y_i - w x_i)
# $$
# 其中，$\bar{x} = \frac{1}{m}\sum^m_{i=1}x_i$为$x$的均值

# 首先编写求解w的函数，传入的参数就是x和y，都是np.ndarray类型的，或是pd.Series类型的（其实都一样）。我们需要大家在下面完成式(6)和式(7)的求解过程，将计算得到的w和b的值返回

# In[10]:


def get_w(x, y):
    '''
    这个函数是计算模型w的值的函数，
    传入的参数分别是x和y，表示数据与标记
    
    Parameter
    ----------
        x: np.ndarray，pd.Series，传入的特征数据

        y: np.ndarray, pd.Series，对应的标记
    
    Returns
    ----------
        w: float, 模型w的值
    '''
    
    
    # m表示样本的数量
    m = len(x)
    
    # 求x的均值
    x_mean = x.mean() # YOUR CODE HERE
    
    # 求w的分子部分
    numerator = 0.0 # YOUR CODE HERE
    for xi in x:
        numerator += xi**2
    
#    numer
    
    # 求w的分母部分
    denominator = None # YOUR CODE HERE
    
    # 求w
    w = None # YOUR CODE HERE
    
    # 返回w
    return w


# In[11]:


def get_b(x, y, w):
    '''
    这个函数是计算模型b的值的函数，
    传入的参数分别是x, y, w，表示数据，标记以及模型的w值
    
    Parameter
    ----------
        x: np.ndarray，pd.Series，传入的特征数据

        y: np.ndarray, pd.Series，对应的标记
        
        w: np.ndarray, pd.Series，模型w的值
    
    Returns
    ----------
        b: float, 模型b的值
    '''
    # 样本个数
    m = len(x)
    
    # 求b
    b = None # YOUR CODE HERE
    
    # 返回b
    return b


# 下面这个类，就是一个最简单的一元线性回归的类，我们已经帮你实现好了三个方法

# In[ ]:


class myLinearRegression:
    def __init__(self):
        '''
        类的初始化方法，不需要初始化的参数
        这里设置了两个成员变量，用来存储模型w和b的值
        '''
        self.w = None
        self.b = None
    
    def fit(self, x, y):
        '''
        这里需要编写训练的函数，也就是调用模型的fit方法，传入特征x的数据和标记y的数据
        这个方法就可以求解出w和b
        '''
        self.w = get_w(x, y)
        self.b = get_b(x, y, self.w)
        
    def predict(self, x):
        '''
        这是预测的函数，传入特征的数据，返回模型预测的结果
        '''
        if self.w == None or self.b == None:
            print("模型还未训练，请先调用fit方法训练")
            return 
        
        return self.w * x + self.b


# ## 5. 预测

# In[ ]:


# 创建一个模型的实例
model1 = myLinearRegression()

# 使用训练集对模型进行训练，传入训练集的LotArea和标记SalePrice

#print(train_data['LotArea'])
#print(train_data['LostArea'][0])
s = 0
for i in train_data['LotArea']:
    s += i

#s /= len(train_data['LotArea'])
print(s)
print(train_data['LotArea'].sum())

model1.fit(train_data['LotArea'], train_data['SalePrice'])

# 对测试集进行预测，并将结果存储在变量prediction中
prediction1 = model1.predict(test_data['LotArea'])


# ## 6. 性能度量

# 模型训练完成后，还需要在测试集上验证其预测能力，这就需要计算模型的一些性能指标，如MAE和RMSE等。
# 
# $$\tag{8}
# MAE(\hat{y}, y) = \frac{1}{m} \sum^m_{i=1} \vert \hat{y} - y \vert
# $$
# 
# $$\tag{9}
# RMSE(\hat{y}, y) = \sqrt{\frac{1}{m} \sum^m_{i=1} (\hat{y} - y)^2}
# $$
# 其中，$\hat{y}$是模型的预测值，$y$是真值，$m$是样本数

# In[ ]:


def MAE(y_hat, y):
    # 请你完成MAE的计算过程
    # YOUR CODE HERE
    
    pass # 删除掉这句pass


# In[ ]:


def RMSE(y_hat, y):
    # 请你完成RMSE的计算过程
    # YOUR CODE HERE
    
    pass # 删除掉这句pass


# 在此计算出模型在测试集上的MAE与RMSE值

# In[ ]:


mae1 = MAE(prediction1, test_data['SalePrice'])
rmse1 = RMSE(prediction1, test_data['SalePrice'])
print("模型1，特征：LotArea")
print("MAE:", mae1)
print("RMSE:", rmse1)


## ## 7. 模型预测效果可视化
#
## In[ ]:
#
#
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
#
#
## In[ ]:
#
#
#plt.figure(figsize = (16, 6))
#
#plt.subplot(121)
#plt.plot(train_data['LotArea'].values, train_data['SalePrice'].values, '.', label = 'training data')
#plt.plot(train_data['LotArea'].values, model1.predict(train_data['LotArea']), '-', label = 'prediction')
#plt.xlabel("LotArea")
#plt.ylabel('SalePrice')
#plt.title("training set")
#plt.legend()
#
#plt.subplot(122)
#plt.plot(test_data['LotArea'].values, test_data['SalePrice'].values, '.', label = 'testing data')
#plt.plot(test_data['LotArea'].values, prediction1, '-', label = 'prediction')
#plt.xlabel("LotArea")
#plt.ylabel('SalePrice')
#plt.title("testing set")
#plt.legend()
#
#
## ### 通过左右两图的对比，分析该模型出现的问题，并给出能帮助模型更好的做预测的方案(选做)
## ###### 双击此处展开讨论
## 
## 
## 
## 
## 
#
## # 使用BsmtUnfSF作为特征，完成模型的训练，指标计算，可视化
#
## In[ ]:
#
#
## YOUR CODE HERE
#
#
#
#
## # 使用GarageArea作为特征，完成模型的训练，指标计算，可视化
#
## In[ ]:
#
#
## YOUR CODE HERE
#
#
#
#
## # 选做：剔除训练集中的离群值(outlier)，然后重新训练模型，观察模型预测性能的变化
## ###### 提示：可以使用下面的代码处理数据
#
## In[ ]:
#
#
## YOUR CODE HERE
#t = train_data[(train_data['LotArea'] < 60000) & (train_data['LotArea'] > 0)] # 将训练集中LotArea小于60000的值存入t
#t = t[t['SalePrice'] < 500000] # 将t中SalePrice小于500000的值保留
#
## 绘制处理后的数据
#plt.figure(figsize = (8, 7))
#plt.plot(t['LotArea'], t['SalePrice'], '.')

