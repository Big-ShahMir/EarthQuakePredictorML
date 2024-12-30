# Earthquake Prediction with Machine Learning

## Overview

This project leverages machine learning techniques to predict earthquake magnitudes and depths based on historical data. By preprocessing and analyzing geographic and temporal features of earthquakes, the model identifies patterns to make predictions. The project explores various configurations of neural networks using hyperparameter tuning to determine the best-performing model.

---

## Features

- **Dataset Preprocessing**: Converts earthquake records into timestamped features and filters relevant attributes like latitude, longitude, depth, and magnitude.
- **Geographical Visualization**: Displays affected regions on a world map using the Basemap library.
- **Machine Learning Model**:
  - Implements a neural network using Keras.
  - Performs hyperparameter tuning with `GridSearchCV`.
  - Scales input and output data for improved training performance.
- **Prediction Outputs**: Predicts earthquake magnitudes and depths based on historical features.
- **Evaluation Metrics**: Assesses model performance using loss and accuracy metrics.

---

## Prerequisites

Ensure you have the following dependencies installed:

- Python 3.8+
- TensorFlow
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Basemap

You can install the required packages using:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib basemap
```

## Dataset
The dataset is expected to be in a CSV file named data.csv, containing the following columns:

- `Date`: Date of the earthquake.
- `Time`: Time of the earthquake.
- `Latitude`: Latitude of the earthquake epicenter.
- `Longitude`: Longitude of the earthquake epicenter.
- `Depth`: Depth of the earthquake in kilometers.
- `Magnitude`: Magnitude of the earthquake on the Richter scale.

## Preprocessing Steps:
- Combines `Date` and `Time` columns to create a `Timestamp`.
- Filters the dataset to retain only relevant columns.
- Handles invalid or missing data entries.

## Usage

### 1. Clone the Repository
Start by cloning the repository and navigating to the project directory:
```bash
git clone <repository_url>
cd src
```
### 2. Run the Script
Execute the main script to preprocess the data, train the machine learning model, and visualize results:
```bash
python app.py
```
### 3. Output
- Model Parameters: The script will display the best hyperparameters identified through grid search.
- Evaluation Metrics: The test loss and accuracy will be printed to the console after evaluation.
- Geographical Visualization: A map will be generated to show earthquake locations. To display the map, ensure the following line is uncommented in the script:
```python
mlp.show()
```
### 4. Cleanup
To clean up any temporary or generated files, simply remove any artifacts or generated outputs as needed.
```bash
rm -rf <generated_files>
```

## Visualtization 
The project generates a map of earthquake locations:
- Blue Dots: Represent earthquake epicenters.
- Coastlines, Countries, and Continents: Visualized to provide context to the locations.
To view the map, uncomment the following line in the code:
```python
mlp.show()
```

## License
This project is licensed under the MIT License. See `LICENSE` for more details. Feel free to use the code as a basis for further projects or as a learning tool.
