#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement

# ## To Predict which language the given Input text is in.

# In[1]:


#Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import warnings
warnings.simplefilter('ignore')


# In[2]:


#Loading the dataset
data = pd.read_csv('Language Detection.csv')


# In[3]:


data.head(10)


# In[4]:


data.shape


# In[5]:


#value count for each language
data['Language'].value_counts()


# In[6]:


#separating the independent and dependent features
X = data['Text']
y = data['Language']


# In[7]:


#converting categorical variables to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# ## Text pre-processing

# In[8]:


data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)


# ## Bag of Words

# In[9]:


#creating bag of words using count vectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(ngram_range=(1,3),analyzer = 'char')
X = cv.fit_transform(data_list).toarray()


# In[10]:


X.shape


# ## Train-Test Split

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20) 


# ## Model Building

# In[12]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()


# In[13]:


clf.fit(X_train, y_train)


# In[14]:


# prediction 
y_pred = clf.predict(X_test)


# **Evaluating the model**

# In[15]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


# In[16]:


print("Accuracy is :",ac)


# In[17]:


# classification report
print(cr)


# In[18]:


# visualising the confusion matrix
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()


# In[19]:


from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)


# In[20]:


# prediction 
y_pred = model.predict(X_test)


# In[21]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)


# In[22]:


print("Accuracy is :",ac)


# In[23]:


# classification report
print(cr)


# In[24]:


# visualising the confusion matrix
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot = True)
plt.show()


# In[25]:


def predict(text):
     X = cv.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(X) # predicting the language
     lang = le.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0]) # printing the language


# In[26]:


predict('Sachin Ramesh Tendulkar BR (/ˌsʌtʃɪn tɛnˈduːlkər/ (listen); pronounced [sət͡ʃin t̪eːɳɖulkəɾ]; born 24 April 1973) is an Indian former international cricketer who captained the Indian national team. Nicknamed "The Little Master"[4] and "Master Blaster"[5]')


# In[27]:


predict('भेजना चाहते हैं हिंदी में मैसेज लेकिन नहीं आती टाइपिंग? इन आसान Tips से मोबाइल से भेजें हिंदी में टेक्स्ट मैसेज')


# In[28]:


predict('ಡೆವಲಪರ್‌ಗಳು ತಮ್ಮ ಆ್ಯಪ್ ನಿಮ್ಮ ಡೇಟಾವನ್ನು ಹೇಗೆ ಸಂಗ್ರಹಿಸುತ್ತದೆ ಮತ್ತು ಬಳಸುತ್ತದೆ ಎಂಬುದರ ಕುರಿತು ಮಾಹಿತಿಯನ್ನು ಇಲ್ಲಿ ತೋರಿಸಬಹುದು. ಡೇಟಾ ಸುರಕ್ಷತೆಯ ಕುರಿತು ಇನ್ನಷ್ಟು ತಿಳಿಯಿರಿ')


# In[29]:


predict('El Sahara (/səˈhɑːrə/, /səˈhærə/) es un desierto del continente africano. Con un área de 9 200 000 kilómetros cuadrados (3 600 000 millas cuadradas), es el desierto cálido más grande del mundo y el tercer desierto más grande en general, más pequeño que los desiertos de la Antártida y el norte del Ártico.[1][2][3 ]')


# In[30]:


predict('9،200،000 كيلومتر مربع (3،600،000 ميل مربع) ، وهي أكبر صحراء حارة في العالم وثالث أكبر صحراء بشكل عام ، وهي أصغر فقط من صحراء أنتاركتيكا وشمال القطب الشمالي.')


# In[31]:


predict('ആഫ്രിക്കൻ ഭൂഖണ്ഡത്തിലെ ഒരു മരുഭൂമിയാണ് സഹാറ (/səˈhɑːrə/, /səˈhærə/). 9,200,000 ചതുരശ്ര കിലോമീറ്റർ (3,600,000 ചതുരശ്ര മൈൽ) വിസ്തീർണ്ണമുള്ള ഇത് ലോകത്തിലെ ഏറ്റവും വലിയ ചൂടുള്ള മരുഭൂമിയും മൊത്തത്തിൽ മൂന്നാമത്തെ വലിയ മരുഭൂമിയുമാണ്, അന്റാർട്ടിക്കയിലെയും വടക്കൻ ആർട്ടിക്കിലെയും മരുഭൂമികളേക്കാൾ ചെറുതാണ്.[1][2][3 ]')


# In[32]:


import pickle


# In[33]:


pickle.dump(clf, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

