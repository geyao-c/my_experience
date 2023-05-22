#!/usr/bin/env python
# coding: utf-8

# # 数据分析步骤

# ## 数据预处理

#    ### 描述性统计分析

# In[1]:


import pandas as pd
mobile = 'mobile.csv'
data = pd.read_csv(mobile, index_col = 'Customer_ID')
explore = data.describe( include = 'all').T  # T是转置，转置后更方便查阅
display(explore)
explore['null'] = len(data)-explore['count']  # describe()函数自动计算非空值数，需要手动计算空值数

explore = explore[['null', 'max']]
explore.columns = [u'空值数', u'最大值']  # 表头重命名
print ("-----------------------------------------------------------------------------------------")
display(explore)
print(len(data))


# ### 缺失值处理

# In[15]:


import pandas as pd
from scipy.interpolate import lagrange  # 导入拉格朗日插值函数
def ployinterp_column(s, n, k=5):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))]  # 取数
    y = y[y.notnull()]  # 剔除空值
    return lagrange(y.index, list(y))(n)  # 插值并返回插值结果

# 逐个元素判断是否需要插值
for i in data.columns:
    for j in range(len(data)):
        if (data[i].isnull())[j]:  # 如果为空即插值。
            data[i][j] = ployinterp_column(data[i], j)
data.head(10)


# ### 异常值处理

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
plt.figure(figsize=(10,8))  # 建立图像
p = data.boxplot(return_type='dict')  # 画箱线图，直接使用DataFrame的方法
plt.title(u'异常值处理前的箱线图')
plt.show()  # 展示箱线图
#异常值处理--3sigma原则确定异常值
for x in range(0,6):
    mean = np.mean(data.iloc[:,x], axis=0)
    std = np.std(data.iloc[:,x], axis=0)
    floor = mean - 3*std
    upper = mean + 3*std
    for i, val in enumerate(data.iloc[:,x]):
        data.iloc[:,x][i] = float(np.where(((val<floor)|(val>upper)), mean, val))
plt.figure(figsize=(10,8))  # 建立图像
p = data.boxplot(return_type='dict')  # 画箱线图，直接使用DataFrame的方法
plt.title(u'异常值处理后的箱线图')
plt.show()  # 展示箱线图


# ### 相关性分析

# In[17]:


#相关性矩阵
dt_corr = data.corr(method = 'pearson')
print('相关性矩阵为：\n',dt_corr)

# 绘制热力图
import seaborn as sns
plt.subplots(figsize=(18, 10)) # 设置画面大小 
sns.heatmap(dt_corr, annot=True, vmax=1, square=True, cmap='Blues') 
plt.show()
plt.close


# ### 合并新的数据集

# In[18]:


# 提取属性合并为新的数据集
data[u'工作日电话时长'] = data['Peak_mins']+data['OffPeak_mins']
newdata = data[[u'工作日电话时长','Weekend_mins','International_mins','average_mins','Total_mins']]
newdata.columns = [u'工作日电话时长',u'周末电话时长',u'国际电话时长',u'平均电话时长',u'总电话时长']
newdata.head()


# ### 标准化处理

# In[30]:


'''from sklearn.preprocessing import StandardScaler
data1 = StandardScaler().fit_transform(newdata)
pd.DataFrame(data1).head()


# ## 数据模型分析：K-Mean 聚类分析

# ### 确定聚类数目

# In[31]:


#肘部法测确定聚类数目

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist  # 计算距离时
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 正常显示中文字体
font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=10) 

K = range(1, 10)
meandistortions = []

for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data1)
    meandistortions.append(sum(np.min(cdist(data1, kmeans.cluster_centers_, 'euclidean'), axis=1)) / data1.shape[0])

plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel(u'平均畸变程度',fontproperties=font)
plt.title(u'用肘部法则来确定最佳的K值',fontproperties=font);
plt.show()


# ### 开始聚类

# In[7]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans  # 导入kmeans算法

k = 4  # 确定聚类中心数

# 构建模型，随机种子设为123
kmeans_model = KMeans(n_clusters = k,n_jobs=4,random_state=123)
fit_kmeans = kmeans_model.fit(data1)  # 模型训练

# 查看聚类结果
kmeans_cc = kmeans_model.cluster_centers_  # 聚类中心
print('各类聚类中心为：\n',kmeans_cc)
kmeans_labels = kmeans_model.labels_  # 样本的类别标签
print('各样本的类别标签为：\n',kmeans_labels)
r1 = pd.Series(kmeans_model.labels_).value_counts()  # 统计不同类别样本的数目
print('最终每个类别的数目为：\n',r1)
# 输出聚类分群的结果
cluster_center = pd.DataFrame(kmeans_model.cluster_centers_,                           columns = [u'工作日电话时长',u'周末电话时长',u'国际电话时长',u'平均电话时长',u'总电话时长']  )# 将聚类中心放在数据框中
cluster_center.index = pd.DataFrame(kmeans_model.labels_ ).                  drop_duplicates().iloc[:,0]  # 将样本类别作为数据框索引
display(cluster_center)


# ### 模型评估

# In[8]:


#模型评估
from sklearn import metrics
s = metrics.silhouette_score(data1,kmeans_model.labels_,metric='euclidean') # 轮廓系数得分
print(s)


# ### 结果可视化——雷达图

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
# 客户分群雷达图
labels = [u'工作日电话时长',u'周末电话时长',u'国际电话时长',u'平均电话时长',u'总电话时长']
legen = ['客户群' + str(i + 1) for i in [3,0,2,1]] # 客户群命名，作为雷达图的图例
lstype = ['-','--',(0, (3, 5, 1, 5, 1, 5)),':','-.']
kinds = list(cluster_center.iloc[:, 0])
print (kinds)
# 由于雷达图要保证数据闭合，因此再添加L列，并转换为 np.ndarray
cluster_center = pd.concat([cluster_center, cluster_center[[u'工作日电话时长']]], axis=1)
centers = np.array(cluster_center.iloc[:, 0:])
print (centers)
# 分割圆周长，并让其闭合
n = len(labels)
angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
angle = np.concatenate((angle, [angle[0]]))
print (angle)
# 绘图
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, polar=True)  # 以极坐标的形式绘制图形
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 
# 画线
for i in range(len(kinds)):
    ax.plot(angle, centers[i], linestyle=lstype[i], linewidth=2, label=kinds[i])
# 添加属性标签
ax.set_thetagrids(angle[:-1] * 180 / np.pi, labels)
plt.title('客户特征分析雷达图')
plt.legend(legen)
plt.show()
plt.close()
