#!/usr/bin/env python



# load datasets package from scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import RandomForestClassifier
import seaborn as sns
import scipy.stats as stats
import train_test_split
import RandomForestClassifier
import accuracy_score, classification_report
import cross_val_score
import RandomForestClassifier
import GridSearchCV
import StandardScaler
import MLPClassifier
import accuracy_score
import StandardScaler
import MLPClassifier
import accuracy_score
import MinMaxScaler
import MLPClassifier
import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MeanShift
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, silhouette_score
import tensorflow as tf
from tensorflow.keras import layers, model
from sklearn.preprocessing 
from sklearn.neural_network 
from sklearn.metrics 
from sklearn.preprocessing 
from sklearn.neural_network 
from sklearn.metrics 
from sklearn.neural_network 
from sklearn.metrics 
from sklearn.preprocessing 
from sklearn.ensemble 
from sklearn.model_selection 
from sklearn.model_selection 
from sklearn.ensemble 
from sklearn.metrics 
from sklearn.model_selection
from sklearn import datasets
from skimpy import skim
from sklearn.ensemble 

#from sklearn datasets, the desired dataset is loaded

mydata= datasets.load_wine()

mydata


#This step checks for clases in the dataset
print("The 3 classes of the wine dataset are : ",mydata['target_names'])

# extraction of features and labels
X = mydata.data
y = mydata.target

# Printing the shape of the data

print("Shape of labels: ", y.shape)
print("Shape of features: ", X.shape)
#print("Names of the features: ",mydata['feature_names'])
print(mydata.DESCR)

# Convert the dataset to a pandas DataFrame
#mywine_dataset = pd.DataFrame(mydata.data, columns=mydata.feature_names)

# Convert the dataset to a pandas DataFrame
mywine_dataset = pd.DataFrame(mydata.data, columns=mydata.feature_names)
mywine_dataset['target'] = mydata.target

# Display the DataFrame
print("Converted DataFrame:")
print(mywine_dataset.head())
#To generate summary statistics of mywine_dataset
skim(mywine_dataset)
#Data pre-processing  
#conduct a check for missing values 
missing_values = mywine_dataset.isna().sum()
print("Missing Values:")
print(missing_values)
#below results indicates no missing values in the dataset, in your report, state how you would have handled it if there was missing data
mywine_dataset.info()

# Count the number of occurrences of each target class
class_counts = mywine_dataset['target'].value_counts()

# Plot a bar chart of the class distribution
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('target Class')
plt.ylabel('Count')
plt.title('Distribution of target Variable')
plt.show()

#  histogram representation  of each feature and normal distribution probability density function
mywine_dataset.hist(bins=12, figsize=(22,17))
plt.suptitle("Histogram Representation of Wine Dataset Features", fontsize=18, y=0.99)

for ax in plt.gcf().get_axes():
    print(ax.get_title())
    feature_name = ax.get_title().split()[0]
    x_min, x_max = ax.get_xlim()
    x_axis = np.linspace(x_min, x_max, 100)
    mean = mywine_dataset[feature_name].mean()
    std_dev = mywine_dataset[feature_name].std()
    y_axis = stats.norm.pdf(x_axis, mean, std_dev) * ax.get_ylim()[1]
    ax.plot(x_axis, y_axis, 'r', linewidth=2)
    ax.set_title(feature_name.capitalize() + " Histogram")
    
plt.show()

# create boxplots of features with target variable
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))

for i, column in enumerate(mywine_dataset.columns[:-1]):
    ax = axes[i // 4, i % 4]
    sns.boxplot(x="target", y=column, data=mywine_dataset, ax=ax)
    ax.set_xlabel("target")
    ax.set_ylabel(column)
    
plt.tight_layout()
plt.show()

#Calculate the correlation matrix
corr = mywine_dataset.corr()

#Create a figure and set its size
plt.figure(figsize=(12, 8))

#Plot the heatmap with annotations
sns.heatmap(corr, annot=True, cmap='BuPu')

#Set the title of the plot
plt.title(' myWine_dataset variables using Pearson correlation coefficient ')

#Display the plot
plt.show()

#part of Data Pre processing

#Data Scaling and Splitting
scaler=StandardScaler()
scaler.fit(X)
xscaled=scaler.transform(X)
X_scaled=pd.DataFrame(xscaled,columns=mywine_dataset.columns[:-1])
X_scaled.head()

# Create an instance of the StandardScaler
#scaler = StandardScaler()

# Fit the scaler to the features and transform the features
#X_scaled = scaler.fit_transform(X)

# Print the scaled features
#print("Scaled features:")
#print(X_scaled)

# #TASK 1 - Implenting the Random Forest Model

X = mywine_dataset.drop('target', axis=1)
y = mywine_dataset['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
print(y)

rfc = RandomForestClassifier()
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
# Instantiate and train the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rfc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1-score
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Perform cross-validation
cv_scores = cross_val_score(rfc, X_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Average Cross-validation score:", cv_scores.mean())

# Define the Random Forest classifier
rf = RandomForestClassifier()

# Defining  the hyperparameter grid to be used
param_grid = {
    'n_estimators': [50,100, 150,],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# to Perform grid search cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and scores (Results)
print("My Best hyperparameters grid: ", grid_search.best_params_)
print("My Train accuracy: ", grid_search.best_score_)
print(" My Test accuracy: ", grid_search.best_estimator_.score(X_test, y_test))


# #TASK 2 MLP
# Define MLP classifier parameters
hidden_layer_sizes = (20, 15, 10)
activation = 'logistic'
random_state = 2
max_iter = 1000

# Creation of  MLP classifier object
mlp_classifier = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    random_state=random_state,
    max_iter=max_iter
)

# Train the MLP classifier model
mlp_classifier.fit(X_train, y_train)

# Prediction of the labels for the test set
y_pred_mlp = mlp_classifier.predict(X_test)

#  The accuracy of the MLP classifier calculation 
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print("The MLP Classifier Accuracy:", accuracy_mlp)


# #MLP Using StandardScaler

# Create StandardScaler object
scaler = StandardScaler()

# Perform data normalization
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define MLP classifier parameters
hidden_layer_sizes = (20, 15, 10)
activation = 'logistic'
random_state = 2
max_iter = 1000

# Create MLP classifier object
mlp_classifier_normalized = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    random_state=random_state,
    max_iter=max_iter
)

# Train the MLP classifier on the normalized data
mlp_classifier_normalized.fit(X_train_normalized, y_train)

# Predict labels for normalized test set
y_pred_normalized = mlp_classifier_normalized.predict(X_test_normalized)

# Calculate accuracy on normalized test set
accuracy_normalized = accuracy_score(y_test, y_pred_normalized)

print("Normalized MLP Classifier Accuracy:", accuracy_normalized)
# Creation of  MinMaxScaler object
scaler = MinMaxScaler()

#  to Perform the data normalization
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define MLP classifier parameters
hidden_layer_sizes = (20, 15, 10)
activation = 'logistic'
random_state = 2
max_iter = 1000

# The Creation of  MLP classifier object
mlp_classifier_normalized = MLPClassifier(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=activation,
    random_state=random_state,
    max_iter=max_iter
)

# Train the MLP classifier on the normalized data
mlp_classifier_normalized.fit(X_train_normalized, y_train)

# Prediction labels for normalized test set
y_pred_normalized = mlp_classifier_normalized.predict(X_test_normalized)

# Calculation of  accuracy on normalized test set
accuracy_normalized = accuracy_score(y_test, y_pred_normalized)

print("The Normalized MLP Classifier Accuracy:", accuracy_normalized)

# #comparism between RF and MLP

# Define the Random Forest classifier
rf = RandomForestClassifier()

# Define the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(20, 15, 10), activation='logistic', random_state=2, max_iter=1000)

# Train the Random Forest classifier
rf.fit(X_train, y_train)

# Train the MLP classifier
mlp.fit(X_train, y_train)

# Predict labels for test set using Random Forest
y_pred_rf = rf.predict(X_test)

# Predict labels for test set using MLP
y_pred_mlp = mlp.predict(X_test)

# Calculate accuracy for Random Forest
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Calculate accuracy for MLP
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

# Print the accuracy for Random Forest
print("Random Forest Classifier Accuracy:", accuracy_rf)

# Print the accuracy for MLP
print("MLP Classifier Accuracy:", accuracy_mlp)

# Define the hyperparameter grid for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform grid search cross-validation for Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5)
grid_search_rf.fit(X_train, y_train)

# Print the best hyperparameters and scores for Random Forest
print("Random Forest Best Hyperparameters: ", grid_search_rf.best_params_)
print("Random Forest Train Accuracy: ", grid_search_rf.best_score_)
print("Random Forest Test Accuracy: ", grid_search_rf.best_estimator_.score(X_test, y_test))

# Define the hyperparameter grid for MLP
param_grid_mlp = {
    'hidden_layer_sizes': [(20, 15, 10), (10, 10), (5, 5, 5)],
    'activation': ['logistic', 'relu'],
    'max_iter': [500, 1000, 1500]
}

# Perform grid search cross-validation for MLP
grid_search_mlp = GridSearchCV(estimator=mlp, param_grid=param_grid_mlp, cv=5)
grid_search_mlp.fit(X_train, y_train)

# Print the best hyperparameters and scores for MLP
print("MLP Best Hyperparameters: ", grid_search_mlp.best_params_)
print("MLP Train Accuracy: ", grid_search_mlp.best_score_)
print("MLP Test Accuracy: ", grid_search_mlp.best_estimator_.score(X_test, y_test))


# #using bagging as an ensemble method 

# Define MLP classifier parameters
mlp_params = {
    'hidden_layer_sizes': (20, 15, 10),
    'activation': 'logistic',
    'random_state': 2,
    'max_iter': 1000
}

# Create MLP classifier object
mlp_classifier = MLPClassifier(**mlp_params)

# Create Bagging classifier with MLP as base estimator
bagging_classifier = BaggingClassifier(base_estimator=mlp_classifier, n_estimators=10, random_state=2)

# Train the Bagging classifier model
bagging_classifier.fit(X_train, y_train)

# Predict labels for the test set
y_pred_bagging = bagging_classifier.predict(X_test)

# Calculate the accuracy of the Bagging classifier
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

print("The Bagging MLP Classifier Accuracy:", accuracy_bagging)


# #TASK 3 Deep Convolutional Neural Network
# #(CNN) 

# Creation of  StandardScaler object
scaler = StandardScaler()

# Performing  data normalization
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Reshape the input data for CNN (assuming X_train/X_test are 2D arrays)
X_train_reshaped = X_train_normalized.reshape(X_train_normalized.shape[0], X_train_normalized.shape[1], 1)
X_test_reshaped = X_test_normalized.reshape(X_test_normalized.shape[0], X_test_normalized.shape[1], 1)

# Define the CNN architecture
model = models.Sequential()
model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_normalized.shape[1], 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Compillation of  the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training  the CNN model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluating the model on the test set
_, accuracy_normalized = model.evaluate(X_test_reshaped, y_test)
print("Normalized CNN Accuracy:", accuracy_normalized)

# Getting the test loss and accuracy from the training history
test_loss = history.history['val_loss'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


#Adjustment of the features in CNN Model above
# Create StandardScaler object
scaler = StandardScaler()

# Perform data normalization
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Reshape the input data for CNN (assuming X_train/X_test are 2D arrays)
X_train_reshaped = X_train_normalized.reshape(X_train_normalized.shape[0], X_train_normalized.shape[1], 1)
X_test_reshaped = X_test_normalized.reshape(X_test_normalized.shape[0], X_test_normalized.shape[1], 1)

# Define the CNN architecture
model = models.Sequential()
model.add(layers.Conv1D(64, kernel_size=5, activation='relu', input_shape=(X_train_normalized.shape[1], 1)))
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = model.fit(X_train_reshaped, y_train, epochs=20, batch_size=64, validation_data=(X_test_reshaped, y_test))

# Evaluate the model on the test set
_, accuracy_normalized = model.evaluate(X_test_reshaped, y_test)
print("Normalized CNN Accuracy:", accuracy_normalized)

# Get the test loss and accuracy from the training history
test_loss = history.history['val_loss'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# #advanced technique using LSTM (Long short-term memory)
# Create StandardScaler object
scaler = StandardScaler()
lkj0
# Perform data normalization
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Reshape the input data for LSTM (assuming X_train/X_test are 2D arrays)
X_train_reshaped = X_train_normalized.reshape(X_train_normalized.shape[0], X_train_normalized.shape[1], 1)
X_test_reshaped = X_test_normalized.reshape(X_test_normalized.shape[0], X_test_normalized.shape[1], 1)

# Define the LSTM architecture
model = models.Sequential()
model.add(layers.LSTM(64, input_shape=(X_train_normalized.shape[1], 1)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluate the model on the test set
_, accuracy_normalized = model.evaluate(X_test_reshaped, y_test)
print("Normalized LSTM Accuracy:", accuracy_normalized)

# Get the test loss and accuracy from the training history
test_loss = history.history['val_loss'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# #hyperparameter optimization


# Create StandardScaler object
scaler = StandardScaler()

# Perform data normalization
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Reshape the input data for CNN (assuming X_train/X_test are 2D arrays)
X_train_reshaped = X_train_normalized.reshape(X_train_normalized.shape[0], X_train_normalized.shape[1], 1)
X_test_reshaped = X_test_normalized.reshape(X_test_normalized.shape[0], X_test_normalized.shape[1], 1)

# Define the CNN architecture
def create_model(hidden_units=64, dropout_rate=0.2):
    model = models.Sequential()
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train_normalized.shape[1], 1)))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden_units, activation='relu'))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=create_model, verbose=0)

# Define the hyperparameters to search
param_grid = {
    'hidden_units': [32, 64, 128],
    'dropout_rate': [0.2, 0.3, 0.4],
    'epochs': [10],
    'batch_size': [32, 64]
}

# Perform grid search cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
grid_result = grid_search.fit(X_train_reshaped, y_train)

# Print the best parameters and accuracy
print("Best Parameters: ", grid_result.best_params_)
print("Best Accuracy: ", grid_result.best_score_)


# Evaluate the best model on the test set
best_model = grid_result.best_estimator_
accuracy = best_model.score(X_test_reshaped, y_test)
print("Test Accuracy: ", accuracy)


# #advanced activation function

# In[84]:



# Creation of StandardScaler object
scaler = StandardScaler()

# Performing data normalization
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Reshape the input data for CNN (assuming X_train/X_test are 2D arrays)
X_train_reshaped = X_train_normalized.reshape(X_train_normalized.shape[0], X_train_normalized.shape[1], 1)
X_test_reshaped = X_test_normalized.reshape(X_test_normalized.shape[0], X_test_normalized.shape[1], 1)

# Define the CNN architecture with Leaky ReLU activation
model = models.Sequential()
model.add(layers.Conv1D(32, kernel_size=3, input_shape=(X_train_normalized.shape[1], 1)))
model.add(layers.LeakyReLU(alpha=0.2))  # Leaky ReLU activation
model.add(layers.MaxPooling1D(pool_size=2))
model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.LeakyReLU(alpha=0.2))  # Leaky ReLU activation
model.add(layers.Dense(3, activation='softmax'))

# Compilation of the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the CNN model
history = model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, validation_data=(X_test_reshaped, y_test))

# Evaluating the model on the test set
_, accuracy_normalized = model.evaluate(X_test_reshaped, y_test)
print("Normalized CNN Accuracy:", accuracy_normalized)

# Getting the test loss and accuracy from the training history
test_loss = history.history['val_loss'][-1]
test_accuracy = history.history['val_accuracy'][-1]
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# #Task 4 K-means clustering

# Creation of StandardScaler object
scaler = StandardScaler()

# Performing data normalization
X_normalized = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=2)
cluster_labels = kmeans.fit_predict(X_normalized)

# Evaluate the accuracy of the clusters
cluster_accuracy = accuracy_score(y, cluster_labels)
print("Cluster Accuracy:", cluster_accuracy)


# #Determining the optimum number of clusters

# Creation of StandardScaler object
scaler = StandardScaler()

# Performing data normalization
X_normalized = scaler.fit_transform(X)

# Initialize empty lists to store the number of clusters and their corresponding inertia
num_clusters = []
inertia = []

# Iterate over a range of possible cluster numbers
for k in range(2, 11):
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=k, random_state=2)
    cluster_labels = kmeans.fit_predict(X_normalized)
    
    # Append the number of clusters and the corresponding inertia to the lists
    num_clusters.append(k)
    inertia.append(kmeans.inertia_)
    
# Plotting the elbow curve
plt.plot(num_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()

# Calculate the differences between consecutive inertia values
inertia_diff = np.diff(inertia)

# Calculate the second differences
second_diff = np.diff(inertia_diff)

# Find the index corresponding to the elbow point
elbow_index = np.where(second_diff < 0)[0][0] + 1

# Get the optimum number of clusters
optimal_clusters = num_clusters[elbow_index]

print("Optimum Number of Clusters:", optimal_clusters)


# #extra feature ---silhouette analysis



# Creation of StandardScaler object
scaler = StandardScaler()

# Performing data normalization
X_normalized = scaler.fit_transform(X)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=2)
cluster_labels = kmeans.fit_predict(X_normalized)

# Evaluate the accuracy of the clusters
cluster_accuracy = accuracy_score(y, cluster_labels)
print("Cluster Accuracy:", cluster_accuracy)

# Calculate the silhouette score
silhouette_avg = silhouette_score(X_normalized, cluster_labels)
print("Silhouette Score:", silhouette_avg)


# #Agglomerative Clustering (Hierarchical clustering)

# Creation of StandardScaler object
scaler = StandardScaler()

# Performing data normalization
X_normalized = scaler.fit_transform(X)

# Apply Agglomerative Clustering
agg_cluster = AgglomerativeClustering(n_clusters=3)
cluster_labels = agg_cluster.fit_predict(X_normalized)

# Evaluate the accuracy of the clusters
cluster_accuracy = accuracy_score(y, cluster_labels)
print("Cluster Accuracy:", cluster_accuracy)


# #Mean Shift Clustering

# Creation of StandardScaler object
scaler = StandardScaler()

# Performing data normalization
X_normalized = scaler.fit_transform(X)

# Apply Mean Shift clustering
meanshift = MeanShift()
cluster_labels = meanshift.fit_predict(X_normalized)

# Evaluate the accuracy of the clusters
cluster_accuracy = accuracy_score(y, cluster_labels)
print("Cluster Accuracy:", cluster_accuracy)





