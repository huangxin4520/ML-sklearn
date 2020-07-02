#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection  import train_test_split
from sklearn.ensemble import AdaBoostClassifier
import time


# # 加载数据集

# In[2]:


iris = load_iris(as_frame=True)
x=iris.data
y=iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.5)


# In[3]:


x_train.head()


# In[4]:


y_train.head()


# # AdaBoost模型建立

# In[5]:


help(AdaBoostClassifier)


# In[6]:


#参数解释：
clf = AdaBoostClassifier(
    base_estimator=None,
    n_estimators=1000,
    learning_rate=0.01,
    algorithm = 'SAMME.R',
    random_state=2020,
    )

t1=time.time()
clf=clf.fit(x_train, y_train)
pred=clf.predict(x_test)
print('predict:',pred)
score=clf.score(x_test,y_test)
print('score:',score)
t2=time.time()


# # 时间成本

# In[7]:


print('time:',t2-t1)


# In[ ]:




