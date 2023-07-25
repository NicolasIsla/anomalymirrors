# Anomaly Detection in Astronomical Mirrors


## Team Members
- Ignacio Albornoz Alfaro
- Tomás Aguirre M.
- Nicolas Isla
- Jordan Perez

## Introduction
The Anomaly Detection in Astronomical Mirrors project aims to develop a robust system for identifying
and detecting anomalies in astronomical mirrors used in space observation telescopes. 
In this project, we utilize an autoencoder for anomaly detection, a type of neural network specifically 
designed for unsupervised learning tasks.

## Example.ipynb - Data Extraction, Processing, and Training Pipeline for Sensor "m1-3"

Data extraction is carried out using the function *filter_and_save_data*. The data is collected and filtered in VLTi mode, and then saved for all telescopes.

### Step 1: Data Extraction

This happen with the fuction *filter_and_save_data* where the data is recolectada y filtrada en el modo vlti, donde posteriormente es guardada, esto para
todos los telescopios

### Step 2: Data Preprocessing

In this case, the data undergoes a logarithm transformation to ensure that all values are in the same scale.

### Step 3: Model Training

Both the training and the model are implemented in *utils.py*, where you only need to call the function to generate the training.

## Anomaly_detectuin.ipynb 
Once the models are trained, this Jupyter Notebook can be used to apply anomaly detection for each individual sensor 
and sensor combinations, identifying possible anomalies.

### Step 1: Loading Trained Models
Before starting the anomaly detection process, the notebook loads the pre-trained. 

### Step 2: Data Preprocessing
The data for each sensor and sensor combination is preprocessed to ensure it aligns with the input requirements of the trained models.
Logarithmic scaling is employed to transform the raw data into a logarithmic scale

### Step 3: Anomaly Detection
Using the loaded models, the notebook applies anomaly detection algorithms to the preprocessed data. The models identify patterns that deviate significantly from the expected normal behavior and flag them as potential anomalies.

### Step 4: Results Visualization
The detected anomalies are visualized, allowing analysts to observe and investigate potential outliers and irregularities in the data. Visualization techniques may include time series plots and scatter plots.




