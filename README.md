# Automated K-Means Clustering Engine
Try it out live at: https://kmeans-clustering-app-407428159583.europe-west3.run.app/  

https://github.com/user-attachments/assets/9a87214f-1a2b-4eef-80a2-905343c06d7c  

## Introduction
A fully automated K-Means clustering pipeline in the form of an easy-to-use web application.  
All you have to do is input your data in the form of an Excel-file and click a few times. Then your data will be clustered into a specified number of clusters, each of which will be analyzed, visualized, and displayed to you. You also get the choice of downloading your clustered dataset, if you wish to play around with it yourself.  

The goal of this project was to make clustering accessible to anyone. Achieving this required building a standardized pipeline capable of processing any type of tabular data, regardless of its content. Furthermore, it required building and deploying an application that could run the pipeline, making it truly accessible and usable for all.

## Table of Contents
- [Introduction](#introduction)
- [App Usage Guide](#app-usage-guide)
  - [Step 1 - Data Upload](#step-1---data-upload)
  - [Step 2 - Data Overview (/view_data)](#step-2---data-overview-view_data)
  - [Step 3 - Define Order for Multinomial Columns (/define_ordinals)](#step-3---define-order-for-multinomial-columns-define_ordinals)
  - [Step 4 - Choose Number of Clusters (/choose_n_clusters)](#step-4---choose-number-of-clusters-choose_n_clusters)
  - [Step 5 - Cluster Analysis Results (/cluster_analysis)](#step-5---cluster-analysis-results-cluster_analysis)
- [Pipeline Explanation](#pipeline-explanation)
  - [Variable Type Identification](#variable-type-identification)
  - [Categorical Order Definition](#categorical-order-definition)
  - [Binary and One Hot Encoding](#binary-and-one-hot-encoding)
  - [Feature Scaling and Dimensionality Reduction](#feature-scaling-and-dimensionality-reduction)
  - [Silhouette Analysis](#silhouette-analysis)
  - [K-Means Clustering](#k-means-clustering)
  - [Cluster Analysis](#cluster-analysis)
- [Deployment on GCP Cloud Run](#deployment-on-gcp-cloud-run)
  - [Docker file](#docker-file)
  - [CI/CD Setup](#cicd-setup)
- [Technologies](#technologies)


## App Usage Guide
The usage of the app is fairly straightforward but for the sake of documentation, we will go through the process step by step.
If you don't have a dataset available, but still wish to try out the app, you can download one of the datasets from the /datasets folder from this repository.

### Step 1 - Data Upload:  
The dataset you're uploading must be an Excel file that follows the following guidelines.
- Column names in the first row  
- Columns must only contain one datatype   
- No empty fields  
- Columns containing categorical data must be of string (text) datatype in order to be identified and treated as categorical.

### Step 2 - Data Overview (/view_data):  
Here you will get the columns of your uploaded data displayed together with a definition of whether they are numerical or categorical.  
If there are categorical columns with more than two categories (multinomial columns) you get to decide if any of them have a specific order you wish to define (ordinal columns).  
![image](https://github.com/user-attachments/assets/0e3bb803-e1db-49b6-a894-9e4246a43c55)  
If not, choose "No" and you will skip step 3 and directly proceed to step 4.

### Step 3 - Define Order for Multinomial Columns (/define_ordinals):  
All your multinomial columns will be displayed on this page. You only have to define a specific order, for the columns where it is relevant. You can leave the rest of the columns blank.
For the columns you do wish to define a specific order for, you need to enter the order of each category from lowest to highest. See example below:  
![image](https://github.com/user-attachments/assets/3260ceec-ab0d-487d-acbb-c9880d251056)  

### Step 4 - Choose Number of Clusters (/choose_n_clusters):  
A silhouette analysis has been run in the background, which has determined the optimal number of clusters for your dataset. You now have the option to proceed with the recommended number or define your own desired number of clusters in the input field. If you wish to proceed with the recommended number of clusters, then leave the input field blank and proceed.

### Step 5 - Cluster Analysis Results (/cluster_analysis):  
This is the final step of the process where you get to see the visualizations of the analysis of each cluster.  
Each cluster analysis contains three elements:  
-Table of descriptive statistics of numerical variables  
-Histograms of the distributions in the numerical variables  
-Pie charts of distributions in the categorical variables  

You also have the choice to download the clustered dataset as an Excel file. This is basically the same data you uploaded in the beginning, but with an extra column defining the cluster label for each row.

## Pipeline Explanation  
In this section, we will go through the different steps of the pipeline and what is happening underneath the hood in each step.
The whole pipeline is visualized below:
![image](https://github.com/user-attachments/assets/99da10f3-418e-4d9e-ae55-e1c1cf764c59)  

### Variable Type Identification  
Before any preprocessing can happen, we need to identify the data type of each column, which will become the variables for the K-Means model later on.
This part of the pipeline is taken care of by the three functions 'identify_dtypes', 'category_counter', and 'get_multinomial_categories'.  

'identify_dtypes' identifies the columns as either categorical or numerical.  
If there are any categorical columns, the 'category_counter' function will count how many categories they contain.  
If any of the categorical columns have more than two categories, they will be identified by the 'get_multinomial_categories' function

### Categorical Order Definition  
On the /view_data page, the user has to choose whether or not any of the identified multinomial columns have a specific order. If the answer here is 'yes', the user can define the desired order for the columns where it is relevant, on the next page. The desired order will then be applied to its column by the 'apply_order_mapping' function. This means that the strings in the column will be replaced by the appropriate integers, essentially applying appropriate encoding.
```python
# Function to apply order mapping to each column if mapping is defined
def apply_order_mapping(raw_df, column_name, mapping):
    if mapping != 'N/A':
        raw_df[column_name] = raw_df[column_name].map(mapping)
    return raw_df
```

### Binary and One Hot Encoding
Since K-Means models can only take numerical inputs, all categorical columns must be encoded, whether or not they have a specific order. The 'apply_one_hot_encoding' function checks if there is defined a category order for that column. If not, it will one-hot-encode the column by creating dummy variables for each category. 

```python
# Function to one-hot-encode columns and create dummy variables
def apply_one_hot_encoding(raw_df, column_name, order_mappings):
    if order_mappings[column_name] == 'N/A':
        # One-hot-encode the column
        dummies = pd.get_dummies(raw_df[column_name], prefix=column_name, drop_first=True)
        raw_df = pd.concat([raw_df, dummies], axis=1)
        raw_df.drop(column_name, axis=1, inplace=True)
    return raw_df
```  

The rest of the columns, which at this point of the process, should only be binary categorical columns are encoded by the 'apply_binary_mapping' function. 
  
```python
# Function to map binary object columns to numeric 
def apply_binary_mapping(raw_df):
    # Identify the remaining object columns (binary categorical columns)
    object_columns = raw_df.select_dtypes(include='object').columns
    for col in object_columns:
        # Get unique values
        unique_values = raw_df[col].unique()
        # Create mapping
        mapping = {unique_values[0]: 0, unique_values[1]: 1}
        # Apply mapping
        raw_df[col] = raw_df[col].map(mapping)
    return raw_df
```

### Feature Scaling and Dimensionality Reduction  
Scaling is a technique that is done in order to prevent bias in the K-Means model. Bias can occur if numerical variables in the same dataset have different scales. For example, a variable that is ranging between 1 to 500 will contribute more to the model, than a variable that is ranging between 1 to 10. Therefore, scaling is performed to standardize all numerical variables into one common scale, ensuring that all variables contribute equally to the model. This technique is applied on the dataset using the 'scale_encoded_df' function. 

```python
# Function to scale the encoded dataframe
def scale_encoded_df(encoded_df):
    # Initialize the scaler
    scaler = StandardScaler()
    # Scale all columns in the encoded dataframe
    scaled_df = pd.DataFrame(scaler.fit_transform(encoded_df), columns=encoded_df.columns)
    return scaled_df
```
When trying to create a K-Means pipeline that should work on most datasets, regardless of the number of variables in them, I had to take into account the possibility of users inputting datasets with large amounts of variables. My main concerns were that large amounts of variables in an ML-model could cause overfitting and also decrease the computational efficiency leading to slower processing time. Therefore, I decided to include dimensionality reduction in the form of PCA (Principal Component Analysis) into the pipeline.  

```python
# Function to perform dimensionality reduction using PCA
def reduce_dimensions(scaled_df):
    # Fit PCA and determine the number of components for 95% variance
    pca = PCA()
    pca.fit(scaled_df)

    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # Determine the number of components required to explain 95% of the variance
    num_components = np.where(cumulative_variance >= 0.95)[0][0] + 1
    # Apply PCA with the determined number of components
    pca = PCA(n_components=num_components)
    reduced_data = pca.fit_transform(scaled_df)
    
    # Convert the reduced data to a dataframe
    reduced_df = pd.DataFrame(reduced_data)
    
    # Return dataframe with reduced dimensions
    return reduced_df
```

That way I could make sure that the application captured as much of the variance as possible, in less amount of variables, while keeping the most important information. Also, including the PCA into the pipeline helped decrease the processing time, which would lead to a better user experience.  

### Silhouette Analysis  
On the /choose_n_clusters page, the user gets recommended an optimal number of clusters for the inputted dataset. That recommendation is based on a silhouette analysis done by the 'silhouette_analysis' function.  

```python
Function to determine the optimal number of clusters by using a silhoutte analysis
def silhouette_analysis(reduced_df, max_clusters=10):
    # Empty list to store silhoutte scores 
    silhouette_scores = []
    # Loop through each number of possible clusters
    for n_clusters in range(2, max_clusters + 1):
        # Initialize K-means model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        # Fit the model on the reduced data and get the cluster labels
        cluster_labels = kmeans.fit_predict(reduced_df)
        # Calculate silhoutte score and append it to the list
        silhouette_avg = silhouette_score(reduced_df, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Exctract the position of the highest score and add 2 to it.
    # 2 is added because of the indexing starting at 0 and the number of clusters analysed starts from 2
    optimal_clusters = np.argmax(silhouette_scores) + 2 
    
    return optimal_clusters
```
The user can choose to proceed with this recommended number of clusters or specify their own desired number.  

### K-Means Clustering  
After the desired number of clusters has been chosen, the K-Means model can finally be performed. This happens in the 'perform_KMeans' function.  
```python
# Function to perform K-means clustering on the reduced data and assign those
# clusters to the original dataframe
def perform_KMeans(reduced_df, original_df, clusters):
    # Initialize K-means model
    kmeans = KMeans(n_clusters = clusters)
    # Fit the model on the reduced data and get the cluster labels
    cluster_labels = kmeans.fit_predict(reduced_df) + 1 # 1 is added because of indexing starting from 0
    # Add cluster labels to the original dataframe
    original_df['Cluster'] = cluster_labels
    
    return original_df
```
The function also assigns the cluster labels to the original dataframe, to perform the cluster analysis in the 'perform_cluster_analysis' function, and to make the dataframe downloadable with the assigned labels attached to the original data.  

### Cluster Analysis  
As the final step of the pipeline, a basic cluster analysis is done by the 'perform_cluster_analysis' that in itself utilizes three different functions for each part of the analysis.  
```python
# Function to perform analyses of each cluster and store the results
def perform_cluster_analysis(column_types_df, original_df):
    # Empty list to store cluster analyses for each cluster
    cluster_analysis_list = []
    # Count number of clusters
    n_clusters = original_df['Cluster'].nunique()

    # Loop through all clusters
    for cluster_number in range(1, n_clusters + 1):
        
        # Filter out single cluster from the rest of the original dataframe
        cluster_df = original_df[original_df['Cluster'] == cluster_number]
        
        # Get descriptive statistics
        num_cols_descriptive_stats = get_descriptive_stats(column_types_df, cluster_df)
        # Plot histograms of numerical columns
        histograms_list = plot_numeric_histograms(column_types_df, cluster_df)
        # Plot piecharts of categorical columns
        pie_charts_list = plot_pie_charts(column_types_df, cluster_df)
        
        # Store the full analysis in a dictionary
        single_cluster_analysis = {'Descriptive Statistics': num_cols_descriptive_stats,
                                   'Histograms': histograms_list,
                                   'Pie Charts': pie_charts_list}
        
        # Append the dictionary to the list
        cluster_analysis_list.append(single_cluster_analysis)
        
    # Return the list containing all cluster analyses    
    return cluster_analysis_list
```

'get_descriptive_stats' calculates the descriptive statistics of all the numerical variables.  
'plot_numeric_histograms' plots histograms for all the numerical variables.  
'plot_pie_charts' plots pie charts for all the categorical variables.  
Both 'plot_numeric_histograms' and 'plot_pie_charts' functions utilizes the 'fig_to_base64' function to convert the matplotlib visualizations to base64, which can be embedded in the HTML in order to be displayed for the user. 

## Deployment on GCP Cloud Run  
For deployment of the app, I used Cloud Run. The choice fell on Cloud Run mainly because of the following reasons:  
- Support for filesystem-based sessions in Flask:  
Cloud Run allows the usage of 'filesystem' as the session type in the Flask app configuration. 
```python
app.config['SESSION_TYPE'] = 'filesystem'  # Define the session type to use filesystem
```
This wasn't possible on App Engine, which was another option I initially considered.  

- Docker-based deployment:  
Cloud Run has the option to deploy with Docker file, which provides flexibility in defining the environment

- CI/CD integration:
Cloud Run provides an easy way to configure a CI/CD pipeline directly with my GitHub repository.

### Docker file  
I used the following Docker file to deploy the app:  
```bash
# docker build -t clustering_app .
# docker run --rm -e PORT=3000 -p 8080:8080 --env-file=.env clustering_app

FROM python:3.12-alpine

# Install the required system packages for building Python packages and lscpu utility
RUN apk update && apk add --no-cache \
    gcc \
    g++ \
    musl-dev \
    util-linux  # Install lscpu system utility for parallel processing in scikit-learn

WORKDIR /src

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip python -m pip install -r requirements.txt

COPY . .

EXPOSE 8080

ENTRYPOINT ["gunicorn", "-b", ":8080", "app:app", "--timeout", "300", "--workers", "5"]
```
### CI/CD Setup
The CI/CD was configured to automatically redeploy on pushes to the 'prod' branch of my repository. This setup saves time and simplifies the redeployment process, compared to having to manually redeploy every time, after every change to the app's codebase.  
![image](https://github.com/user-attachments/assets/52145507-27ae-4f0e-9058-b286bace3480)  

## Technologies
- Python: Backend development.
- HTML, CSS, JavaScript: Frontend design and interactivity.
- Scikit-Learn: Machine learning, scaling, PCA (dimensionality reduction), and silhouette analysis
- Flask: Web app development and routing
- Google Cloud Run: Deployment platform for running the application in a scalable and containerized environment
- Google Cloud Secret Manager: Secure storage and management of session secret key
- Docker: Containerization for isolating and deploying the app with all its dependencies
- xlsxwriter: Writing Pandas DataFrames to Excel files for data export
- Pandas and NumPy: Data manipulation and analysis
- Matplotlib: Data visualization
