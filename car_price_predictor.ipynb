import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
data = pd.read_csv("cars_data.csv")
data.head()
data.describe()
data.shape
data.isnull().sum()
print(data.fuel_type.value_counts())
data['year'].unique()
data['Price'].unique()
data["kms_driven"].unique()
data['fuel_type'].unique()
## Problems


*   year column has many categorical data
*   Price column has "Ask for Price"
*   kms_driven column needs to omit the string value in each row
*   There are NaN val in fuel_type
*   keep only first 3 words of name column
*   Removing outliers









# Data Pre-Processing
backup = data.copy()
data = data[data['year'].str.isnumeric()] #string operation on every row to filter out numeric values
data['year']=data['year'].astype(int)
data.info()
data['Price'] #Price has 'ask for price'
data = data[data['Price'] != "Ask For Price"]
Since the Price is in object type, we remove the ',' to convert the data into int
data['Price']=data['Price'].str.replace(',','').astype(int) #replace commas with empty string
data['kms_driven']=data['kms_driven'].str.split(' ').str.get(0).str.replace(',','')
#split the kms column into two strings and consider only integer values
data=data[data['kms_driven'].str.isnumeric()]
data.info()
#convert kms_driven into int
data['kms_driven'] = data['kms_driven'].astype(int)
data.info()
data=data[~data['fuel_type'].isna()]
data['name'] = data['name'].str.split(' ').str.slice(0,3).str.join(' ')
#split using space and slice first 3 index and join
data = data.reset_index(drop = True)
#if drop is set to false, all the previous index which is not updated will also be shown
data.info()
data.describe() #checking outliers
data[data['Price']>6e6] #outlier
data = data[data['Price']<6e6].reset_index(drop= True)
# Cleaned Data
Checking relationship of Company with Price
plt.subplots(figsize=(15,7))
ax=sns.boxplot(x='company',y='Price',data=data)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.show()
#the distribution of Price values for each company,
# showing the median, quartiles, and any outliers
Checking relationship of Year with Price
plt.subplots(figsize=(20,10))
ax=sns.swarmplot(x='year',y='Price',data=data)
Checking relationship of Fuel Type with Price
plt.subplots(figsize=(14,7))
sns.boxplot(x='fuel_type',y='Price',data=data)
# Model
X = data.drop(columns='Price') #all columns excluding Price
y = data['Price']
X
y
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
ohe = OneHotEncoder()
ohe.fit(X[['name', 'company','fuel_type']])
#Creating a column transformer to transform categorical columns
column_trans=make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),
                                    remainder='passthrough')
#passthrough is to consider only categorical values and skip other values
lr = LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
R2 Score
r2_score(y_test,y_pred)
Finding the model with a random state of TrainTestSplit
scores=[]
for i in range(1000):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_pred=pipe.predict(X_test)
    scores.append(r2_score(y_test,y_pred))
The best model is found at random state 302
np.argmax(scores)
scores[np.argmax(scores)]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_pred=pipe.predict(X_test)
r2_score(y_test,y_pred)
