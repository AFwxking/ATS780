#Script for ATS780 Machine Learning for Atmospheric Sciences HW1
#Script takes data prepared from GFS_download.py and CLAVR_x_organizer.py to run Random Forecst model

#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import glob
import xarray as xr
import pandas as pd

# %%

# Specify the local directory where the interpolated GFS data resides
processed_GFS_directory = '/mnt/data2/mking/ATS780/processed_GFS_files/'

# Specify the local directory where the interpolated CLAVRx data resides
clavrx_directory = '/mnt/data2/mking/ATS780/CLAVRX_data/'

#Get the sorted file list in each directory
clavrx_flist = sorted(glob.glob(clavrx_directory + '*'))
GFS_flist = sorted(glob.glob(processed_GFS_directory + '*'))

# %%
#Load data clavrx data and update values to 0 and 1
clavrx_load = xr.open_dataset(clavrx_flist[0])
cloud_mask_data = np.squeeze(clavrx_load['cloud_mask'].data) #0 clear, 1 probably clear, 2 probably cloud, 3 cloudy
cloud_mask = np.empty(cloud_mask_data.shape)
cloud_mask[(cloud_mask_data >= 2 )] = 1 #Anything probably cloudy and cloudy becomes 1
cloud_mask[(cloud_mask_data < 2)] = 0 #Anything probably clear and clear becomes 0

#Load GFS data 
GFS_load = xr.open_dataset(GFS_flist[0])
isobaric = GFS_load['isobaric'].data
relative_humidity_data = np.squeeze(GFS_load['relative_humidity'].data)
vertical_velocity_data = np.squeeze(GFS_load['vertical_velocity'].data)
temperature_data = np.squeeze(GFS_load['temperature'].data)
absolute_vorticity_data = np.squeeze(GFS_load['absolute vorticity'].data)

# Initialize an empty dictionary to store the data for each variable
data_dict = {}

# Variable names
variable_names = ['Cld_Msk', 'RH', 'VV', 'Temp', 'AbsVort']  

# Loop through variable names
for variable in variable_names:

    #Add Cld_Msk values        
    if variable == 'Cld_Msk':
        data = cloud_mask[:, :]

        # Create column name
        column_name = f'{variable}'
        
        # Add data to the dictionary
        data_dict[column_name] = data.flatten()

    # Loop through pressure levels
    for pressure_level in isobaric:
        # Create column name
        column_name = f'{variable}_{pressure_level}mb'
        
        # Extract data for the current variable and pressure level
        if variable == 'RH':
            data = relative_humidity_data[isobaric == pressure_level, :, :]

            # Add data to the dictionary
            data_dict[column_name] = data.flatten()

        elif variable == 'VV':
            data = vertical_velocity_data[isobaric == pressure_level, :, :]

            # Add data to the dictionary
            data_dict[column_name] = data.flatten()

        elif variable == 'VV':
            data = temperature_data[isobaric == pressure_level, :, :]

            # Add data to the dictionary
            data_dict[column_name] = data.flatten()

        elif variable == 'AbsVort':
            data = absolute_vorticity_data[isobaric == pressure_level, :, :]

            # Add data to the dictionary
            data_dict[column_name] = data.flatten()

df = pd.DataFrame(data_dict)

#%%
# Split the data
X = df.drop(columns=['Cld_Msk'])
y = df['Cld_Msk']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=10, random_state=42)

# Define the number of iterations to print progress
progress_interval = 10 

# Training loop
for i in range(0, len(X_train), progress_interval):
    end = min(i + progress_interval, len(X_train))
    X_batch = X_train[i:end]
    y_batch = y_train[i:end]
    
    # Train on the current batch
    rf_classifier.fit(X_batch, y_batch)
    
    # Calculate accuracy on the entire training set (you can use a smaller validation set)
    y_train_pred = rf_classifier.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    print(f"Processed {end}/{len(X_train)} samples. Train Accuracy: {train_accuracy:.2f}")

# %% 
# Evaluate the model
y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# %%
