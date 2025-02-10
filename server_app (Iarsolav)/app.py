
from flask import Flask, render_template, request,send_from_directory, jsonify
import uuid
import plotly.express as px
import pandas as pd
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Exopredict-Machine-Learning-Iaroslav")))

import rank_stars
import rank_planets5
import nn_exo_planets

STAR_DATASET_FILE_PATH = '../Output/CSV_Files/Cleaned Dataset.csv'
OUTPUT_DIR = 'static/charts/'
MODEL_DIR = '../Exopredict-Machine-Learning-Iaroslav'

MODEL = None
SCALER = None

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_stars_from_csv():
    stars = rank_stars.load_dataset_and_extract_star_names(STAR_DATASET_FILE_PATH)
    return stars

# Function to generate a plot based on the selected star
def generate_star_html(selected_star):
    X_normalized, df = rank_stars.load_and_normalize_dataset(file_path=STAR_DATASET_FILE_PATH, features=['Star_Temperature_K', 'Star_Radius_Solar', 'Star_Mass_Solar'])
    rankings = rank_stars.rank_by_similarity(X_normalized, selected_star, df)
    fig = rank_stars.plotly_similarity_ranking_obj(rankings, selected_star)
    # Generate a unique filename for the chart HTML
    chart_filename = f"chart_{uuid.uuid4().hex}.html"
    chart_path = os.path.join(OUTPUT_DIR, chart_filename)
    
    # Save the chart as an HTML file
    fig.write_html(chart_path)
    
    return chart_filename

def extract_host_stars():
    return rank_stars.extract_host_stars(STAR_DATASET_FILE_PATH)

def generate_sphere_data():
    df = pd.read_csv(STAR_DATASET_FILE_PATH)
    ra = df['ra'].apply(np.radians)
    dec = df['dec'].apply(np.radians)

    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)

    scale = 1000.0  
    positions = np.column_stack((x, y, z)) * scale

    unique_stars = df['Host_Star'].unique()
    return jsonify({
        'stars': unique_stars.tolist(),
        'positions': positions.tolist()
    })
    
def convert_ra_to_rad(ra_str):
    # RA is given in hours, minutes, seconds; convert it to radians
    ra_parts = ra_str.split('h')
    ra_deg = float(ra_parts[0]) * 15  # 1 hour = 15 degrees
    return np.radians(ra_deg)

if MODEL is None or SCALER is None:
    MODEL, SCALER = nn_exo_planets.load_model(os.path.join(MODEL_DIR, 'nn_exo_planet_model.keras'),
                                              os.path.join(MODEL_DIR, 'X_scaler.pkl'))

app = Flask(__name__)

@app.route('/')
def home():
    host_stars = extract_host_stars()
    return render_template('index.html', host_stars=host_stars)

@app.route('/rank-star')
def rank_star():
    stars = extract_stars_from_csv()
    return render_template('rank_star.html', stars=stars)

@app.route('/generate-chart')
def generate_chart():
    selected_star = request.args.get('star')
    chart_filename = generate_star_html(selected_star)
    return jsonify({'chart_filename': chart_filename})

@app.route('/view-chart/<filename>')
def view_chart(filename):
    return render_template('view_chart.html', chart_filename=filename)

@app.route('/charts/<filename>')
def serve_chart(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/sphere-data')
def sphere_data():
    # Load your CSV file
    df = pd.read_csv(STAR_DATASET_FILE_PATH)
    # Extract relevant columns
    df = df[['Host_Star', 'rastr', 'dec']].dropna()

    # Convert RA and Dec from degrees to radians
    df['ra_rad'] = df['rastr'].apply(lambda x: convert_ra_to_rad(x))
    df['dec_rad'] = df['dec'].apply(np.radians)  # Dec is already in degrees, so just convert to radians

    # Convert spherical coordinates (RA, Dec) to 3D Cartesian coordinates (x, y, z)
    df['x'] = np.cos(df['dec_rad']) * np.cos(df['ra_rad'])
    df['y'] = np.cos(df['dec_rad']) * np.sin(df['ra_rad'])
    df['z'] = np.sin(df['dec_rad'])

    # Prepare data for plotting
    stars = df['Host_Star'].tolist()
    x_coords = df['x'].tolist()
    y_coords = df['y'].tolist()
    z_coords = df['z'].tolist()

    return render_template('sphere_data.html', stars=stars, x_coords=x_coords, y_coords=y_coords, z_coords=z_coords)

# Prediction page route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the form
        temp = float(request.form['temperature'])
        radius = float(request.form['radius'])
        mass = float(request.form['mass'])
        metallicity = float(request.form['metallicity'])
        
        input_data = np.array([[temp, radius, mass, metallicity]])
        
        # Run model to predict 
        prediction = nn_exo_planets.use_model_predict(MODEL, SCALER, input_data)
               
        # Return the result as part of the response
        return render_template('predict.html', prediction=prediction)
    
    return render_template('predict.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
