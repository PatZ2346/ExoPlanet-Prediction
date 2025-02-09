
from flask import Flask, render_template, request,send_from_directory, jsonify
import uuid
import plotly.express as px

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "machine_analysis")))

import rank_stars
import rank_planets5

STAR_DATASET_FILE_PATH = '../CSV_Files/Cleaned Dataset.csv'
OUTPUT_DIR = 'static/charts/'

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

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)
