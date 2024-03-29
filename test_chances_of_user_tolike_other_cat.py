import pandas as pd

input_file_name=r"C:\Users\user\.spyder-py3\users.csv"
# set output file name and desired number of lines
output_file_name = r"C:\Users\user\.spyder-py3\VidCatt.csv"
df = pd.read_csv(input_file_name, sep=';', encoding='latin1',   engine='python')
category_id = {'Education': 57239, 'Chatting': 319622, 'Art': 417240, 'Travel': 198667,
               'Food': 34715, 'News': 11791, 'DIY': 96634, 'Gaming': 104958,
               'Sport': 16467, 'Beauty': 23302, 'Technology': 45026, 'Commerce': 83806}

# create dataframe from dictionary
df1 = pd.DataFrame.from_dict(category_id, orient='index', columns=['category_id'])

# reset the index to create a new column for category
df1 = df1.reset_index()

# rename columns to category and id
df1.columns = ['category', 'category_id']

# print the resulting DataFrame
from sklearn.utils import shuffle
df1= shuffle(df1)
print(df1)
df_dup = pd.concat([df1]*72370, ignore_index=True)
print(df_dup)

df_dup.to_csv(output_file_name, sep=';', index=False, encoding='latin1', )   

import random 
import pandas as pd

# generate a list of random user ids
user_ids = [random.randint(12470, 100000) for i in range(806671)]

# create a DataFrame from the list of user ids
df = pd.DataFrame({'userID': user_ids})

# save the DataFrame to a CSV file
df.to_csv('output_filee.csv', sep=';', index=False, encoding='latin1')


 
#Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# input
x = df.iloc[:, 0].values
from sklearn.impute import SimpleImputer

# Replace NaN values with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x = imputer.fit_transform(x.reshape(-1, 1))
print(x)
# output
y = df.iloc[:, 5].values
y_reshaped = y.reshape(1, -1) 
print(y)
#Splitting The Dataset: Train and Test dataset
 
  
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=0)

# Reshape y array to 2D
ytrain = ytrain.reshape(-1, 1)
ytest = ytest.reshape(-1, 1)

# Standardize the data
sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

# Print the first 10 rows of xtrain
print(xtrain[0:10, :])
print(x.shape)
print(y.shape)

 
#train 
from sklearn.linear_model import LogisticRegression
  
classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)

#evaluation metrics
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)


from sklearn.metrics import accuracy_score
  
print ("Accuracy : ", accuracy_score(ytest, y_pred))


from matplotlib.colors import ListedColormap

X_set, y_set = xtest, ytest
# access the column by its index instead of hardcoding index 1
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, 
                               stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 0].min() - 1, 
                               stop = X_set[:, 0].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(
             np.array([X1.ravel(), X2.ravel()]).T).reshape(
             X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
  
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
  
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
      
plt.title('Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()




