from flask import Flask, render_template, request, session, redirect, url_for, send_file
from flask_session import Session
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for rendering in-memory (no GUI)
import matplotlib.pyplot as plt
import io
import base64
import xlsxwriter
import os
from google.cloud import secretmanager
import json

# $env:FLASK_APP = "app.py"
# $env:FLASK_ENV = "development"
# python -m flask run

# Flask constructor
app = Flask(__name__)

# Function to pull secrets from Google Secret Manager
def get_secret(secret_id):
    sm_client = secretmanager.SecretManagerServiceClient()
    secret_name = sm_client.secret_path('sylvan-mode-413619', secret_id) + '/versions/1'
    secret_string = sm_client.access_secret_version(name=secret_name).payload.data.decode('utf-8')
    
    try:
        # Try to load the secret as JSON
        return json.loads(secret_string)
    except json.JSONDecodeError:
        # If it fails, return the secret string as is
        return secret_string

app.config['SECRET_KEY'] = get_secret('CLUSTERING_APP_SESSION_SECRET_KEY') # Secret key added for session management and security 
app.config['SESSION_TYPE'] = 'filesystem'  # Define the session type to use filesystem
app.config['SESSION_PERMANENT'] = False  # Optional: Whether the session is permanent
app.config['SESSION_USE_SIGNER'] = True  # Optional: Sign the session cookie for extra security

Session(app)

@app.route("/")
def upload():
    try:
        return render_template('index.html')
    
    # Return error template 
    except Exception as e:
        return render_template('error.html', error_message=str(e))

@app.route('/view_data', methods=['POST', 'GET'])
def view_data():
    try:
        # Clear old session data from possible previous sessions
        session.clear()
        
        # Read the file using Flask request
        file = request.files['file']

        # Parse the data as a dataframe
        raw_df = pd.read_excel(file)
        
        # Identify categorical(ordinal and multinomial) columns and store in dataframes
        column_types_df = identify_dtypes(raw_df)
        column_types_df = category_counter(column_types_df, raw_df)
        multinomial_columns_df = get_multinomial_categories(column_types_df, raw_df)
        
        # Store dataframes in session for later use
        session['raw_df'] = raw_df.to_dict(orient='list')
        session['original_df'] = raw_df.to_dict(orient='list')
        session['column_types_df'] = column_types_df.to_dict(orient='list')

        # If multinomial_columns_df is not empty (there are multinomial columns in the dataset):
        if not multinomial_columns_df.empty:
            # Store multinomial_columns_df in session for later use
            session['multinomial_columns_df'] = multinomial_columns_df.to_dict(orient='list')
            # Store column_types_df in session for later use
            session['column_types_df'] = column_types_df.to_dict(orient='list')
            # Return HTML snippet that will render the table
            return render_template('view_data.html', 
                                   column_types_df=column_types_df.to_html(),
                                   multinomial_columns_df=multinomial_columns_df.to_html())
        else:
            # Store column_types_df in session for later use
            session['column_types_df'] = column_types_df.to_dict(orient='list')
            # Return HTML snippet that will render the table
            return render_template('view_data.html', 
                                   column_types_df=column_types_df.to_html())
    
    # Return error template 
    except Exception as e:
        return render_template('error.html', error_message=str(e))

#______________________________________________________________________________

@app.route('/define_ordinals', methods=['POST', 'GET'])
def define_ordinals():
    try:
        # Get choice from the form
        order_choice = request.form.get('order_choice')

        # Retrieve multinomial_columns_df from the session
        multinomial_columns_df = pd.DataFrame(session.get('multinomial_columns_df'))

        if order_choice == 'no':
            # If no order is chosen just return the default mappings
            order_mappings = {column: 'N/A' for column in multinomial_columns_df['Column Name']}
            # Store order_mappings in session for later use
            session['order_mappings'] = order_mappings
            # Proceed to next step or final processing
            return redirect(url_for('choose_n_clusters'))

        elif order_choice == 'yes':
            # If yes, render the form for specifying order
            return render_template('define_ordinals.html', multinomial_columns_df=multinomial_columns_df)
    
    # Return error template 
    except Exception as e:
        return render_template('error.html', error_message=str(e))

#______________________________________________________________________________

@app.route('/choose_n_clusters', methods=['POST', 'GET'])
def choose_n_clusters():
    try:
        # Retrieve column_types_df from the session
        column_types_df = pd.DataFrame(session.get('column_types_df'))
        # Retrieve raw_df from the session
        raw_df = pd.DataFrame(session.get('raw_df'))
        # Check if order mappings are already set in the session
        order_mappings = session.get('order_mappings', None)

        # If there are categorical columns in the dataset:
        if column_types_df['Variable Type'].str.contains('categorical').any():
            
            # If order_mappings is empty, it means the user chose "yes" and provided orders
            if order_mappings is None:
                # Retrieve multinomial_columns_df from the session
                multinomial_columns_df = pd.DataFrame(session.get('multinomial_columns_df'))

                # Empty dict for order mappings
                order_mappings = {}
                # Loop through each multinomial column
                for column in multinomial_columns_df['Column Name']:
                    # Empty dict for order mapping for a single column
                    order = {}
                    # Get the specified order for each category in the column
                    for category in multinomial_columns_df[multinomial_columns_df['Column Name'] == column]['Categories'].iloc[0]:
                        order_value = request.form.get(f'{column}_order_{category}')
                        # If the order value for the category is specified..  
                        if order_value:
                            # ..store it in the dict
                            order[category] = int(order_value)

                    # If there is specified an order for the categories in the column..
                    if order:
                        # ..add it to order_mappings.. 
                        order_mappings[column] = order
                    else: # ..else assign N/A for the column
                        order_mappings[column] = 'N/A'

            # Apply order mapping and one-hot-encoding for categorical columns
            raw_df = apply_categorical_encoding(raw_df, order_mappings)
            # Delete data from memory
            del order_mappings

            # Map binary columns to numeric 
            encoded_df = apply_binary_mapping(raw_df)
            # Delete data from memory
            del raw_df

            # Scale the encoded dataframe
            scaled_df = scale_encoded_df(encoded_df)
            # Delete data from memory
            del encoded_df
            
            # Perform dimensionality reduction using PCA
            reduced_df = reduce_dimensions(scaled_df)
            # Delete data from memory
            del scaled_df
            
        # If there are only numerical columns in the dataset:
        else:
            # Scale the raw_df dataframe
            scaled_df = scale_encoded_df(raw_df)
            
            # Perform dimensionality reduction using PCA
            reduced_df = reduce_dimensions(scaled_df)
            # Delete data from memory
            del scaled_df

        # Store reduced_df in session for later use
        session['reduced_df'] = reduced_df.to_dict(orient='list')

        # Determine the optimal number of clusters by using a silhouette analysis
        optimal_clusters = silhouette_analysis(reduced_df)
        # Store optimal_clusters in session for later use
        session['optimal_clusters'] = optimal_clusters

        return render_template('choose_n_clusters.html', optimal_clusters=optimal_clusters)
    
    # Return error template 
    except Exception as e:
        return render_template('error.html', error_message=str(e))

#______________________________________________________________________________

@app.route('/cluster_analysis', methods=['POST', 'GET'])
def cluster_analysis():
    try:
        # Retrieve dataframes from the session
        reduced_df = pd.DataFrame(session.get('reduced_df'))
        original_df = pd.DataFrame(session.get('original_df'))
        column_types_df = pd.DataFrame(session.get('column_types_df'))

        # Get the user's choice of clusters or fallback to optimal clusters
        cluster_choice = request.form.get('cluster_choice', session.get('optimal_clusters'))
        cluster_choice = int(cluster_choice) if cluster_choice else session.get('optimal_clusters')

        # Perform KMeans clustering
        clustered_df = perform_KMeans(reduced_df, original_df, cluster_choice)
        
        # Store clustered_df in session to make it downloadable
        session['clustered_df'] = clustered_df

        # Perform cluster analysis
        cluster_analysis_results = perform_cluster_analysis(column_types_df, clustered_df)

        # Pass the analysis results to the template
        return render_template('cluster_results.html', cluster_analysis_results=cluster_analysis_results)
    
    # Return error template 
    except Exception as e:
        return render_template('error.html', error_message=str(e))
    
    
    
@app.route('/download_clustered_df')
def download_clustered_df():
    try:
        # Retrieve the dataframe from the session
        clustered_df = pd.DataFrame(session.get('clustered_df'))
        
        # Save the dataframe to an Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            clustered_df.to_excel(writer, index=False, sheet_name='Clustered Data')
        
        # Go back to the beginning of the BytesIO object
        output.seek(0)
        
        # Send the Excel file as a downloadable file
        return send_file(output, 
                         mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 
                         as_attachment=True, 
                         download_name='clustered_df.xlsx')
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))
#______________________________________________________________________________

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(e):
    return render_template('error.html', error_message="Internal server error"), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))

    
    
###############################################################################
################### Feature Engineering and Preprocessing #####################
###############################################################################

####################### Variable Type Identification ##########################

# Function to specify the variable type of each column as numerical or categorical 
def identify_dtypes(raw_df):
    # Identify original dtypes from raw dataframe
    column_types = raw_df.dtypes
    # Create dataframe holding the dtype for each column
    column_types_df = pd.DataFrame({'Column Name': column_types.index,
                                    'dtype': column_types.values,
                                    'Variable Type': None})
    
    # Loop to specify the variable type as numerical or categorical 
    for dtype in range(len(column_types_df)):
        # If int64 or float64 then assign var_type as numerical
        if column_types_df['dtype'].iloc[dtype] == 'int64' or column_types_df['dtype'].iloc[dtype] == 'float64':
            column_types_df.loc[dtype, 'Variable Type'] = "numerical"
        # If object then assign var_type as categorical
        elif column_types_df['dtype'].iloc[dtype] == 'object':
            column_types_df.loc[dtype, 'Variable Type'] = "categorical"
        else:
            column_types_df.loc[dtype, 'Variable Type'] = "N/A"
    
    # Return dataframe containing column names and variable types
    return column_types_df

#______________________________________________________________________________

# Function to count the number of categories in the categorical columns, 
# thereby defining if its binary or multinomial(multiclass)        
def category_counter(column_types_df, raw_df):
    # Create new column for number of categories
    column_types_df['Number of Categories'] = 'N/A'
    
    # Identify and loop through categorical columns and count categories    
    for column in range(len(column_types_df)):
        # If var_type is categorical..
        if column_types_df['Variable Type'].iloc[column] == 'categorical':
            # ..get the column name..
            column_name = column_types_df['Column Name'].iloc[column]
            # ..and use it to identify it in raw_df to count the categories
            number_of_categories = len(pd.unique(raw_df[column_name]))
            # Then assign the number of categories to the column
            column_types_df.loc[column, 'Number of Categories'] = number_of_categories
            
    # Return column_types_df with the new column of counts of categories
    return column_types_df

#______________________________________________________________________________

# Function to extract all categories from the multinomial variables
def get_multinomial_categories(column_types_df, raw_df):
    # Empty list for appending each row
    multinomial_columns_list = []
    
    # Loop through column_types_df and extract all categories from the multinomial variables
    for column in range(len(column_types_df)):
        # If var_type is categorical..
        if column_types_df['Variable Type'].iloc[column] == 'categorical':
            # ..and it contains more than 2 categories:
            if column_types_df['Number of Categories'].iloc[column] > 2:
                # Get the column name
                column_name = column_types_df['Column Name'].iloc[column]
                # Get all unique categories
                unique_categories = pd.unique(raw_df[column_name]).tolist()
                # Count the number of categories
                num_categories = len(unique_categories)
                
                # Create a new row for multinomial_columns_df
                new_row = {
                    'Column Name': column_name,
                    'Categories': unique_categories,
                    'Number of Categories': num_categories
                }
                
                # Append the new row to the list
                multinomial_columns_list.append(new_row)
                
    # If multinomial_columns_list is not empty:
    if multinomial_columns_list:
        # Create a dataframe from the list of rows
        multinomial_columns_df = pd.DataFrame(multinomial_columns_list)
        # Return multinomial_columns_df that contains categories of each ordinal variable      
        return multinomial_columns_df
    # If multinomial_columns_list is empty:
    else: # Return an empty dataframe
        return pd.DataFrame()

########################### Encoding and Scaling ##############################

# Function to apply order mapping to each column if mapping is defined
def apply_order_mapping(raw_df, column_name, mapping):
    if mapping != 'N/A':
        raw_df[column_name] = raw_df[column_name].map(mapping)
        
    return raw_df

# Function to one-hot-encode columns and create dummy variables
def apply_one_hot_encoding(raw_df, column_name, order_mappings):
    if order_mappings[column_name] == 'N/A':
        # One-hot-encode the column
        dummies = pd.get_dummies(raw_df[column_name], prefix=column_name, drop_first=True)
        raw_df = pd.concat([raw_df, dummies], axis=1)
        raw_df.drop(column_name, axis=1, inplace=True)
        
    return raw_df
    
# Function to apply order mapping and categorical encoding   
def apply_categorical_encoding(raw_df, order_mappings):
    # Apply the appropriate encoding
    for column_name, mapping in order_mappings.items():
        # Encoding for ordinal columns
        raw_df = apply_order_mapping(raw_df, column_name, mapping)
        # Encoding for unordered multinomial columns
        raw_df = apply_one_hot_encoding(raw_df, column_name, order_mappings)    
        
    # Identify boolean columns (that came from one-hot-encoding)
    bool_columns = list(raw_df.select_dtypes(include='bool').columns)
    # Convert all boolean columns to binary (1/0)
    raw_df[bool_columns] = raw_df[bool_columns].astype(int)
    
    return raw_df

#______________________________________________________________________________
    
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
    
#______________________________________________________________________________

# Function to scale the encoded dataframe
def scale_encoded_df(encoded_df):
    # Initialize the scaler
    scaler = StandardScaler()
    # Scale all columns in the encoded dataframe
    scaled_df = pd.DataFrame(scaler.fit_transform(encoded_df), columns=encoded_df.columns)
    return scaled_df
 
###############################################################################
##################### Clustering Pipeline and Optimization ####################
###############################################################################

###################### Dimensionality Reduction using PCA #####################

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

############################# Silhoutte Scoring ###############################

# Function to determine the optimal number of clusters by using a silhoutte analysis
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

############################ K-Means Clustering ###############################

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


###############################################################################
###################### Cluster Analysis and Visualization #####################
###############################################################################

# Function to get descriptive statistics for all numerical columns
def get_descriptive_stats(column_types_df, original_df):
    
    # If there are numerical columns in the dataset:
    if column_types_df['Variable Type'].str.contains('numerical').any():
        # Extract numerical columns
        numeric_columns = column_types_df[column_types_df['Variable Type'] == 'numerical']['Column Name']
        # Create descriptive statistics for all numerical columns
        num_cols_descriptive_stats = original_df[numeric_columns].describe().round(2)
    # If there are only categorical columns in the dataset:
    else: # Create a dummy dataframe with a message
        num_cols_descriptive_stats = pd.DataFrame(['No numerical columns in your dataset'], columns=['data'])
        
    # Return dataframe as HTML containing all descriptive statistics or message
    return num_cols_descriptive_stats.to_html()
    
#______________________________________________________________________________

# Function to convert matplotlib figures to base64 for embedding in HTML
def fig_to_base64(fig):
    # Create an in-memory byte stream to store the image
    img = io.BytesIO()
    # Save the figure into the byte stream, specifying the format, bounding box, and background color
    fig.savefig(img, format='png', bbox_inches='tight', facecolor='#1c1c1c')  # Set facecolor for background
    # Seek to the beginning of the byte stream
    img.seek(0)
    # Encode the byte stream as a base64 string
    base64_img = base64.b64encode(img.getvalue()).decode('utf8')
    # Close the figure to free up memory
    plt.close(fig)
    # Return the base64-encoded image string
    return base64_img


# Function to plot and store histograms for all numerical columns 
def plot_numeric_histograms(column_types_df, original_df):
    # Extract numerical columns
    numeric_columns = column_types_df[column_types_df['Variable Type'] == 'numerical']['Column Name']
    # Empty list to store plot figures
    histograms_list = []
    
    # Create histograms for each numerical column
    for column in numeric_columns:
        # Create a new figure
        fig, ax = plt.subplots()

        # Plot histogram for the column
        ax.hist(original_df[column], 
                bins=20, 
                color='#7c1880',  # Purple color
                edgecolor='black', 
                alpha=0.8)

        # Set background color and labels color
        ax.set_facecolor('#1c1c1c')  # Dark background
        ax.set_title(f'Distribution of {column}', color='white', fontsize=14)
        ax.set_xlabel(column, color='white', fontsize=12)
        ax.set_ylabel('Frequency', color='white', fontsize=12)
        
        # Change tick colors
        ax.tick_params(colors='white')

        # Convert figure to base64 and store it in the list
        histograms_list.append(fig_to_base64(fig))
        
    # Return list containing all histograms for the numerical columns
    return histograms_list

# Function to plot and store pie charts for all categorical columns
def plot_pie_charts(column_types_df, original_df):
    # Extract categorical columns
    categorical_columns = column_types_df[column_types_df['Variable Type'] == 'categorical']['Column Name']
    # Empty list to store pie chart figures
    pie_charts_list = []

    # Create pie charts for each categorical column
    for column in categorical_columns:
        # Calculate the distribution of categories
        category_counts = original_df[column].value_counts()
        
        # Create a new figure
        fig, ax = plt.subplots()
        
        # Plot pie chart for the column
        ax.pie(category_counts, 
               labels=category_counts.index, 
               autopct='%1.1f%%', 
               startangle=90, 
               colors=['#7c1880', '#9c4db7', '#d4a0e0', '#1c1c1c', '#4e4e4e'],  # Custom colors
               textprops={'color':'white'})  # Make text white
            
        # Add title
        ax.set_title(f'Distribution of {column}', color='white', fontsize=14)
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax.axis('equal')

        # Set background color
        fig.patch.set_facecolor('#1c1c1c')
        
        # Convert figure to base64 and store it in the list
        pie_charts_list.append(fig_to_base64(fig))
            
    # Return list containing all pie charts for the categorical columns
    return pie_charts_list

#______________________________________________________________________________

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
        
