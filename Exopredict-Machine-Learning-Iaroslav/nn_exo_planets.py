import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import sklearn as skl
import joblib

def create_and_save_model(file_path='Resources/part-00000-3d57ee90-8dc9-4f89-97e6-768aa0ffce3c-c000.csv', 
                          features=['Star_Temperature_K', 'Star_Radius_Solar', 'Star_Mass_Solar', 'Star_Metallicity'],
                          save_model_name='nn_exo_planet_model.keras',
                          scaler_file_name='X_scaler.pkl'):
    # Load dataset
    df = pd.read_csv(file_path)

    # Group by 'Host_Star' to aggregate planet counts and star features
    star_data = df.groupby('Host_Star').agg({
        'Num_Planets': 'first',
        'Star_Temperature_K': 'first',
        'Star_Radius_Solar': 'first',
        'Star_Mass_Solar': 'first',
        'Star_Metallicity': 'first'
    }).reset_index()

    # Normalize features
    X = star_data[features].values
    y = star_data['Num_Planets'].values

    # Normalize the features
    #X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create scaler instance
    X_scaler = skl.preprocessing.StandardScaler()

    # Fit the scaler
    X_scaler.fit(X_train)

    # Scale the data
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(len(features),)),  # First hidden layer
        Dense(32, activation='relu'),  # Second hidden layer
        Dense(1)  # Output layer for regression
    ])

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse' , metrics=["accuracy"])

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,  # Using part of training data for validation
        epochs=100,
        batch_size=32,
        verbose=1
    )

    model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

    # Save model
    model.save(save_model_name)
    # Save scaler
    joblib.dump(X_scaler, scaler_file_name)  # Save it for future use

def load_model(save_model_name='nn_exo_planet_model.keras', scaler_file_name='X_scaler.pkl'):
    model = tf.keras.models.load_model(save_model_name)
    X_scaler = joblib.load(scaler_file_name)
    return model, X_scaler

def use_model_predict(model, X_scaler, star_data_raw):
    star_data_scaled = X_scaler.transform(star_data_raw)
    prediction = model.predict(star_data_scaled)
    rounded_prediction = round(prediction[0, 0])
    return rounded_prediction

if __name__ == "__main__":
    # create_and_save_model()
    
    # Load the pre-trained model
    model, scaler = load_model()
    model.summary()
    print("Model input shape:", model.input_shape)
    print("Model output shape:",model.output_shape)
    
    star_data_raw  = np.array([[5800, 1.1, 1.05, 0.02]])  # Sun-like star (?)
    prediction = use_model_predict(model, scaler, star_data_raw)
    print("Prediction:", prediction)
    