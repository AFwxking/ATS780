#Script for ATS780 Machine Learning for Atmospheric Sciences HW1
#Script takes data prepared from GFS_download.py and CLAVR_x_organizer.py to run Random Forecst model

#%%
from graphviz import Source # To plot trees
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.tree import export_graphviz
import glob
import xarray as xr
import pandas as pd
import random

# %%

# Specify the local directory where the interpolated GFS data resides
processed_GFS_directory = '/mnt/data2/mking/ATS780/processed_GFS_files/'

# Specify the local directory where the interpolated CLAVRx data resides
clavrx_directory = '/mnt/data2/mking/ATS780/CLAVRX_data/'

# Specify the local direcotry where the persistance interpolated CLAVRx data resides
persistance_directory = '/mnt/data2/mking/ATS780/Persist_CLAVRX_data_/'

#Get the sorted file list in each directory
clavrx_flist = sorted(glob.glob(clavrx_directory + '*'))
GFS_flist = sorted(glob.glob(processed_GFS_directory + '*'))
persist_flist = sorted(glob.glob(persistance_directory + '*'))

#%%
#Select a number of random latitude/longitude values to use on each file

#Define the lat/lon values used from CLAVR_x_organizer
res = 0.02
left_lon = -100
right_lon = -65
top_lat = 50
bottom_lat = 25

#One dimensional arrays defining longitude and latitude
len_lon = np.round(np.arange(left_lon,right_lon, res),2)
len_lat = np.round(np.arange(bottom_lat, top_lat, res),2)

#Us numpy meshgrid function to create 2d coordinates using lat/lon values
meshlon, meshlat = np.meshgrid(len_lon, len_lat)

#Set random seed for reproducibility
random.seed(42)

#Generate random lat/lon pairs
num_of_pairs_each_time = 200
random_lat_lon_idx_pairs = np.empty((len(clavrx_flist)*num_of_pairs_each_time, 2)).astype(int)
for idx in range(len(clavrx_flist)*num_of_pairs_each_time):
    lat_idx = random.randint(0, np.shape(len_lat)[0] - 1)
    lon_idx = random.randint(0, np.shape(len_lon)[0] - 1)
    random_lat_lon_idx_pairs[idx,0] = lat_idx
    random_lat_lon_idx_pairs[idx,1] = lon_idx

# %%

#Loop through files, pull out data based on selected lat/lon indexes and place into dataframe
for idx in range(len(clavrx_flist)):

    #Load clavrx data and update values to 0 and 1
    clavrx_load = xr.open_dataset(clavrx_flist[idx])
    cloud_mask_data = np.squeeze(clavrx_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    cloud_mask = np.empty(cloud_mask_data.shape)
    cloud_mask[(cloud_mask_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    cloud_mask[(cloud_mask_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load clavrx data and update values to 0 and 1
    persist_load = xr.open_dataset(persist_flist[idx])
    persist_data = np.squeeze(persist_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
    persist_mask = np.empty(persist_data.shape)
    persist_mask[(persist_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
    persist_mask[(persist_data < 2)] = 0 #Anything probably clear and clear becomes 0

    #Load GFS data 
    GFS_load = xr.open_dataset(GFS_flist[idx])
    isobaric = GFS_load['isobaric'].data
    relative_humidity_data = np.squeeze(GFS_load['relative_humidity'].data)
    vertical_velocity_data = np.squeeze(GFS_load['vertical_velocity'].data)
    temperature_data = np.squeeze(GFS_load['temperature'].data)
    absolute_vorticity_data = np.squeeze(GFS_load['absolute vorticity'].data)
    cloud_mixing_ratio_data = np.squeeze(GFS_load['cloud_mixing_ratio'].data)
    total_cloud_cover_data = np.squeeze(GFS_load['total_cloud_cover'].data)

    # Initialize an empty dictionary to store the data for each variable
    data_dict = {}

    # Variable names
    variable_names = ['Cld_Msk', 'Cld_Msk_Persist','RH', 'VV', 'Temp', 'AbsVort', 'Cld_Mix_Ratio', 'Total_Cld_Cvr']  

    #Current lat/lon index values
    pair_idx_1 = idx * num_of_pairs_each_time
    pair_idx_2 = (idx * num_of_pairs_each_time) + num_of_pairs_each_time

    # Loop through variable names
    for variable in variable_names:

        #Add Cld_Msk values        
        if variable == 'Cld_Msk':
            
            data = cloud_mask[random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data
        
        #Add Cld_Msk values        
        if variable == 'Cld_Msk_Persist':
            
            data = persist_mask[random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ] ]

            # Create column name
            column_name = f'{variable}'
            
            # Add data to the dictionary
            data_dict[column_name] = data

        # Loop through pressure levels
        for pressure_level in isobaric:
            # Create column name
            column_name = f'{variable}_{pressure_level}mb'
            
            # Extract data for the current variable and pressure level
            if variable == 'RH':
                data = relative_humidity_data[isobaric == pressure_level, random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'VV':
                data = vertical_velocity_data[isobaric == pressure_level, random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Temp':
                data = temperature_data[isobaric == pressure_level, random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'AbsVort':
                data = absolute_vorticity_data[isobaric == pressure_level, random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Cld_Mix_Ratio':
                data = cloud_mixing_ratio_data[isobaric == pressure_level, random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

            elif variable == 'Total_Cld_Cvr':
                data = total_cloud_cover_data[isobaric == pressure_level, random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 0 ], random_lat_lon_idx_pairs[ pair_idx_1 : pair_idx_2 , 1 ]]

                # Add data to the dictionary
                data_dict[column_name] = data

    if idx == 0: #If first file...create dataframe
        df = pd.DataFrame(data_dict)
    else: #If any other...append dataframe to first
        next_df = pd.DataFrame(data_dict)
        df = pd.concat([df, next_df], ignore_index=True)

    print(f'{idx + 1}/{len(clavrx_flist)} completed', end='\r')

#%%

#Save Dataframe
df.to_csv('HW1_data.csv', index=False)

# #Load a DataFrame from a CSV file...run if needed
# df = pd.read_csv('HW1_data.csv')

#%%
# Split the data
X = df.drop(columns=['Cld_Msk','Cld_Msk_Persist'])
y = df[['Cld_Msk','Cld_Msk_Persist']]

# Reserve the held-back testing data
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=.2,random_state=13) 

# Now reserve validation for hyperparamter tuning
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=.2,random_state=13)

# Get Cld_Msk_Persist for each dataset to use as baseline
y_test_baseline = y_test['Cld_Msk_Persist']
y_test = y_test.drop(columns=['Cld_Msk_Persist'])
y_train_baseline = y_train['Cld_Msk_Persist']
y_train = y_train.drop(columns=['Cld_Msk_Persist'])
y_val_baseline = y_val['Cld_Msk_Persist']
y_val = y_val.drop(columns=['Cld_Msk_Persist'])

# %%
#Define random forest and train model

#Define Hyperparameters
fd = {
    "tree_number": 15,    # number of trees to "average" together to create a random forest
    "tree_depth": 8,      # maximum depth allowed for each tree
    "node_split": 50,     # minimum number of training samples needed to split a node
    "leaf_samples": 50,    # minimum number of training samples required to make a leaf node
    "criterion": 'gini',  # information gain metric, 'gini' or 'entropy'
    "bootstrap": False,   # whether to perform "bagging=bootstrap aggregating" or not
    "max_samples": None,  # number of samples to grab when training each tree IF bootstrap=True, otherwise None 
    "random_state": 13    # set random state for reproducibility
}

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators = fd["tree_number"],
                           random_state = fd["random_state"],
                           min_samples_split = fd["node_split"],
                           min_samples_leaf = fd["leaf_samples"],
                           criterion = fd["criterion"],
                           max_depth = fd["tree_depth"],
                           bootstrap = fd["bootstrap"],
                           max_samples = fd["max_samples"])

#Train random forest
rf_classifier.fit(X_train, y_train)

#Make prediction on all training data
y_pred_train = rf_classifier.predict(X_train)

#%%
#Confusion Matrix on training data

acc = metrics.accuracy_score(y_train, y_pred_train)
print("training accuracy: ", np.around(acc*100), '%')

confusion = confusion_matrix(y_train, y_pred_train)

print(confusion)

pred_classes = ['Pred No Cloud', 'Pred Cloud']
true_classes = ['True No Cloud', 'True Cloud']

def confusion_matrix_plot(predclasses, targclasses, pred_classes, true_classes):
  class_names = np.unique(targclasses)
  table = []
  for pred_class in class_names:
    row = []
    for true_class in class_names:
        row.append(100 * np.mean(predclasses[targclasses == true_class] == pred_class))
    table.append(row)
  class_titles_t = true_classes
  class_titles_p = pred_classes
  conf_matrix = pd.DataFrame(table, index=class_titles_t, columns=class_titles_p)
  display(conf_matrix.style.background_gradient(cmap='Greens').format("{:.1f}"))

#Plot Confusion Matrix
confusion_matrix_plot(y_pred_train, y_train['Cld_Msk'], pred_classes, true_classes)

#Plot Confusion Matrix for baseline
confusion_matrix_plot(y_train_baseline, y_train['Cld_Msk'], pred_classes, true_classes)

acc = metrics.accuracy_score(y_train, y_train_baseline)
print("baseline training accuracy: ", np.around(acc*100), '%')

#Confusion numbers for baseline
confusion = confusion_matrix(y_train, y_train_baseline)
print(confusion)

#%%
#Confusion Matrix on validation data

#Make prediction on validation data
y_pred_val = rf_classifier.predict(X_val)

acc = metrics.accuracy_score(y_val, y_pred_val)
print("validation accuracy: ", np.around(acc*100), '%')

confusion_validation = confusion_matrix(y_val, y_pred_val)

print(confusion_validation)

#Plot Confusion Matrix
confusion_matrix_plot(y_pred_val, y_val['Cld_Msk'], pred_classes, true_classes)

#Plot Confusion Matrix for baseline
confusion_matrix_plot(y_val_baseline, y_val['Cld_Msk'], pred_classes, true_classes)

acc = metrics.accuracy_score(y_val, y_val_baseline)
print("baseline validation accuracy: ", np.around(acc*100), '%')

#Confusion numbers for baseline
confusion = confusion_matrix(y_val, y_val_baseline)
print(confusion)

#%%
#Look at individual tree
local_path = '/home/mking/ATS780/'
fig_savename = 'rf_cloud_tree'
tree_to_plot = 0 # Enter the value of the tree that you want to see!

#Get predictor feature names
column_names = X.columns
column_names = column_names.tolist()

tree = rf_classifier[tree_to_plot] # Obtain the tree to plot
tree_numstr = str(tree_to_plot) # Adds the tree number to filename

complete_savename = fig_savename + '_' + tree_numstr + '.dot'
export_graphviz(tree,
                out_file=local_path + '/' + complete_savename,
                filled=True,
                proportion=False,
                leaves_parallel=False,
                feature_names=column_names)

Source.from_file(local_path + complete_savename)

#%%
#Feature importance

def calc_importances(rf, feature_list):
    ''' Calculate feature importance '''
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

    # Print out the feature and importances 
    print('')
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
    print('')

    return importances

def plot_feat_importances(importances, feature_list):
    ''' Plot the feature importance calculated by calc_importances ''' 
    plt.figure(figsize=(19,35))
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.barh(x_values, importances)
    # Tick labels for x axis
    plt.yticks(x_values, feature_list)
    # Axis labels and title
    plt.xlabel('Importance'); plt.ylabel('Variable'); plt.title('Variable Importances')
    
    
plot_feat_importances(calc_importances(rf_classifier, column_names),  column_names)


#%%
#Permutation importance

# Single-pass permutation
permute = permutation_importance(
    rf_classifier, X_val, y_val, n_repeats=20, random_state=fd["random_state"])

# Sort the importances
sorted_idx = permute.importances_mean.argsort()

def plot_perm_importances(permute, sorted_idx, feature_list):
    ''' Plot the permutation importances calculated in previous cell '''
    # Sort the feature list based on 
    new_feature_list = []
    for index in sorted_idx:  
        new_feature_list.append(feature_list[index])

    fig, ax = plt.subplots(figsize = (19,25))
    ax.boxplot(permute.importances[sorted_idx].T,
           vert=False, labels=new_feature_list)
    ax.set_title("Permutation Importances")
    fig.tight_layout()
    
plot_perm_importances(permute, sorted_idx, column_names)


# %% 
# Evaluate the model on test data
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

#Confusion numbers for baseline
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

#Plot Confusion Matrix
confusion_matrix_plot(y_pred, y_test['Cld_Msk'], pred_classes, true_classes)

#Plot Confusion Matrix for baseline
confusion_matrix_plot(y_test_baseline, y_test['Cld_Msk'], pred_classes, true_classes)

acc = metrics.accuracy_score(y_test, y_test_baseline)
print("baseline validation accuracy: ", np.around(acc*100), '%')

#Confusion numbers for baseline
confusion = confusion_matrix(y_test, y_test_baseline)
print(confusion)

# %%
