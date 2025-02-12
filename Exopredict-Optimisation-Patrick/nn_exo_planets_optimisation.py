import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import sklearn as skl
import joblib
import shutil
import json

def create_and_save_model(file_path='ml_iaroslav/Resources/Cleaned Dataset.csv', 
                          output_dir='Exopredict-Optimisation-Patrick',
                          features=['Star_Temperature_K', 'Star_Radius_Solar', 'Star_Mass_Solar', 'Star_Metallicity'],
                          save_model_name='nn_exo_planet_model.keras',
                          scaler_file_name='X_scaler.pkl',
                          results_file_name='results.json'):
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
        Dense(256, activation='relu', input_shape=(len(features),)),  # First hidden layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),  # Second hidden layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),  # Third hidden layer
        BatchNormalization(),
        Dropout(0.3),
        Dense(1)  # Output layer for regression
    ])

    model.summary()

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,  # Using part of training data for validation
        epochs=300,
        batch_size=64,
        verbose=1,
        callbacks=[early_stopping, reduce_lr]
    )

    model_loss, model_accuracy = model.evaluate(X_test_scaled, y_test, verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

    # Save model
    model.save(save_model_name)
    # Save scaler
    joblib.dump(X_scaler, scaler_file_name)  # Save it for future use
    
    # Save outputs to output_dir as well
    shutil.copy(save_model_name, output_dir + '/' + save_model_name)
    shutil.copy(scaler_file_name, output_dir + '/' + scaler_file_name)
    
    np.savez(output_dir + '/train_test_data.npz', 
             X_train=X_train, X_test=X_test, 
             y_train=y_train, y_test=y_test, 
             X_train_scaled=X_train_scaled, X_test_scaled=X_test_scaled)
    
    # Save results to JSON file
    results = {
        "model_loss": model_loss,
        "model_accuracy": model_accuracy,
        "training_history": history.history
    }
    with open(output_dir + '/' + results_file_name, 'w') as f:
        json.dump(results, f)

def load_model(save_model_name='Exopredict-Optimisation-Patrick/nn_exo_planet_model.keras', scaler_file_name='Exopredict-Optimisation-Patrick/X_scaler.pkl'):
    model = tf.keras.models.load_model(save_model_name)
    X_scaler = joblib.load(scaler_file_name)
    return model, X_scaler

def use_model_predict(model, X_scaler, star_data_raw):
    star_data_scaled = X_scaler.transform(star_data_raw)
    prediction = model.predict(star_data_scaled)
    rounded_prediction = round(prediction[0, 0])
    return rounded_prediction

if __name__ == "__main__":
    create_and_save_model()
