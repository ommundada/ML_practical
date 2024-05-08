Assi 1 :

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Specify the path to your CSV file
file_path = '/content/MLdataass1 (1).csv'


# Read the CSV file
df = pd.read_csv(file_path)


# Assuming your columns are named 'height' and 'weight'
X = df[['Height']]
y = df['Weight']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a linear regression model
model = LinearRegression()


# Fit the model to the training data
model.fit(X_train, y_train)


# Make predictions on the test data
predictions = model.predict(X_test)


# Plot the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression: Weight Prediction from Height')
plt.show()


new_heights = [[167], [170], [178]]  # Replace these values with the heights you want to predict
new_predictions = model.predict(new_heights)


# Plot the regression line for the test data
plt.scatter(X_test, y_test, color='black', label='Actual Data')
plt.plot(X_test, predictions, color='blue', linewidth=3, label='Regression Line')
plt.scatter(new_heights, new_predictions, color='red', label='Predicted Data')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression: Weight Prediction from Height')
plt.legend()
plt.show()


# Display the new predictions
for height, prediction in zip(new_heights, new_predictions):
    print(f'Height: {height[0]}, Predicted Weight: {prediction}')


#ASS1- 2nd way-
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

#Load your dat set from a CSV file

#Replace 'your_dataset.csv' with the actual file path

file_path ='/Users/shivzatnakhandare/Desktop/SOCR-Heightweight.csv'

try:

    data=pd.read_csv(file_path)

except FileNotFoundError:

    print(f"File (file_path) not found. Please check the file path.")

 #Add additional handling if needed

    exit()

#Check if the dataset is empty

if data.empty:

    print("The dataset is empty.")

    exit()

 #1. Define dependent and independent variables
X = data[['Height']] # Independent variable

y= data['Weight'] #Dependent variable

 #2. Split the data into train and test sets (80% train, 20% test)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

 #3. Create and train the linear regression model

model= LinearRegression()

model.fit(X_train, y_train)

 #4. Evaluate the model's accuracy (R-squared score)

accuracy= model.score(X_test, y_test)

print(f'Accuracy Score: (accuracy)')

29 #5. Perform prediction

height_to_predict = [[175]] # Example: Predict weight for a person with height 175

predicted_weight =model.predict(height_to_predict) 
print(f'Predicted Weight: {predicted_weight[0]}')

#6. Plot the linear model

34 plt.scatter(X_test, y_test, color='purple')

36 plt.plot(X_test, model.predict(X_test), color='black', linewidth=3)

36 plt.title('Linear Regression Model')

37 plt.xlabel('Height')

38 plt.ylabel('Weight')

39 plt.show()



Assi 2 :

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib.pyplot as plt
data = pd.read_csv("D:\Codes\.vscode\python\mammals.csv")
X=data['brain_wt'] #Independent Variable
y=data['body_wt']  #dependent Variable
print(f"shape of features: {X.shape}")
print(f"shape of labels: {y.shape}")
X = np.array(X).reshape(-1, 1)
data.head()
(X_train,X_test, y_train,y_test)=train_test_split(X, y, test_size=0.2, random_state=40)
print(f"Shape of x train: {X_train.shape}")


print(f"Shape of x test: {X_test.shape}")


print(f" Shape of y train: {y_train.shape}")


print(f"shape of y test: {y_test.shape}")
# Create a linear regression model
model = LinearRegression()
# Train the model on the training set
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred
# Assuming you have trained your model and made predictions
mse = mean_squared_error(X_test, y_pred)
r2 = r2_score(X_test, y_pred)  #accuracy
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='black') # Actual data points
plt.plot(X_test, y_pred, color='blue', linewidth=3) # Linear regression line
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.title('Linear Regression Model')
plt.show()




Assi 3 :

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv('/content/dataset_ml_assignment_3.csv')
data.head()  # Display the first few rows of the dataset

data.shape  # Show the dimensions of the dataset (rows, columns)

data.info()  # Display information about the dataset, including data types and null values

data.isnull().sum()  # Count the number of null values in each column

# Calculate the correlation matrix
cor = data.corr()

# Visualize the correlation matrix using a heatmap
import seaborn as sns
plt.figure(figsize=(15,10))
sns.heatmap(np.abs(cor), cmap="YlGnBu")  # Absolute correlation values to focus on magnitude
plt.title("Correlation Heatmap")

# Calculate correlations of all columns with the target variable 'MEDV' and sort them
correlations = data.corrwith(data['MEDV']).sort_values()

# Assign colors for positive and negative correlations
colors = []
for i in correlations:
    if i < 0:
        colors.append('r')  # Red for negative correlation
    else:
        colors.append('g')  # Green for positive correlation

# Plot bar chart showing correlations
plt.barh(correlations.index, correlations, color=colors)
plt.title('Correlations of All columns with MEDV')
plt.show()

# Split the data into features (X) and target variable (y)
X = data.drop('MEDV', axis=1)
y = data.MEDV

print(X.shape)  # Display the shape of features (X)
print(y.shape)  # Display the shape of target variable (y)

# 2) Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)  # Display the shape of training features
print(X_test.shape)   # Display the shape of testing features
print(y_train.shape)  # Display the shape of training target variable
print(y_test.shape)   # Display the shape of testing target variable

# Initialize and train a Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions using the trained model on the testing set
y_pred = lr.predict(X_test)

# 4) Mean Squared Error (MSE) 
print("MSE = ", mean_squared_error(y_test, y_pred))  # Lower MSE indicates better fit

# 1) R2 score
print("r2_score =", r2_score(y_test, y_pred))  # R2 score closer to 1 indicates better fit

# 5) Display the coefficients and intercept of the linear regression model
print(f"Coefficients: {lr.coef_}\n")
print(f"Intercept: {lr.intercept_}")

# Generate the equation of the linear regression model
eqn = ["y", "=", f"{lr.intercept_}", "+"]  # Initialize the equation with intercept
for i in range(len(lr.coef_)):
    eqn.append(f"({round(lr.coef_[i], 2)}){X.columns[i]}")  # Add each coefficient term
    eqn.append("+")

final_eqn = " ".join(eqn)  # Combine the equation terms into a final string
print(final_eqn)  # Display the final equation

# 6) Plot a scatter plot of actual vs predicted values
plt.figure(figsize=(15,10))
plt.scatter(y_test, y_pred)



Assi 4 : 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.linear_model import LogisticRegression  # Import Logistic Regression model
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting
from sklearn.metrics import accuracy_score  # Import accuracy_score for model evaluation


# Load the dataset from an Excel file
data = pd.read_excel("/content/dataset_ml_assignment_4.xlsx")  # RES 1
data.head()  # Display the first few rows of the dataset


# Count the number of occurrences of each class in the 'class' column
data["class"].value_counts()  # RES 2


# Check for missing values in the dataset
data.isna().sum()  # RES 3


# Separate the target variable 'class' and the features
y = data["class"]  # Target variable
x = data.drop(["class"], axis=1)  # Features


# Split the data into training and testing sets (70% training, 30% testing)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=52)


# Initialize a Logistic Regression model
logreg = LogisticRegression()


# Train the Logistic Regression model on the training data
logreg = logreg.fit(xtrain, ytrain)


# Make predictions on the test data using the trained model
ypred = logreg.predict(xtest)


# Calculate the accuracy of the Logistic Regression model
accuracy_score(ytest, ypred)  # RES 4


from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier


# Create Decision Tree classifier object
clf = DecisionTreeClassifier()


# Train Decision Tree Classifier on the training data
clf = clf.fit(xtrain, ytrain)


# Predict the response for the test dataset using the Decision Tree model
ypred = clf.predict(xtest)


# Calculate the accuracy of the Decision Tree model
print("Accuracy:", accuracy_score(ytest, ypred))  # RES 5




Assi 5 :


# Importing necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Generating synthetic data using make_blobs with 1000 samples, 4 centers, and 2 features
X, _ = make_blobs(n_samples=1000, centers=4, n_features=2)


# Plotting the generated data points
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# Initializing an empty list to store silhouette scores
silhouette_scores = []


# Looping over a range of cluster numbers from 2 to 10
for k in range(2, 11):
    # Instantiating KMeans clustering model with k clusters and 10 initializations
    kmeans = KMeans(n_clusters=k, n_init=10)
    # Fitting the model to the data
    kmeans.fit(X)
    # Calculating silhouette score for the current clustering
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    # Appending the silhouette score to the list
    silhouette_scores.append(silhouette_avg)


# Plotting silhouette scores against the number of clusters
plt.figure(figsize=(5, 3))
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()



Assi 6 :




# Importing necessary libraries
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans


# Generating synthetic data using make_blobs with 1000 samples, 4 centers, and 2 features
X, _ = make_blobs(n_samples=1000, centers=4, n_features=2)


# Plotting the generated data points
plt.scatter(X[:, 0], X[:, 1])
plt.show()


# Initializing an empty list to store within-cluster sum of squares (WCSS)
wcss = []


# Looping over a range of cluster numbers from 2 to 10
for k in range(2, 11):
    # Instantiating KMeans clustering model with k clusters and 10 initializations
    kmeans = KMeans(n_clusters=k, n_init=10)
    # Fitting the model to the data
    kmeans.fit(X)
    # Appending the WCSS value for the current clustering to the list
    wcss.append(kmeans.inertia_)


# Plotting WCSS against the number of clusters
plt.figure(figsize=(5, 3))
plt.plot(range(2, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()


