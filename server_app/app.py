
from flask import Flask, render_template

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "machine_analysis")))

import rank_stars
import rank_planets5

def extract_stars_from_csv():
    file_path='../machine_analysis/Resources/part-00000-3d57ee90-8dc9-4f89-97e6-768aa0ffce3c-c000.csv'
    stars = rank_stars.load_dataset_and_extract_star_names(file_path)
    return stars

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/rank-star')
def rank_star():
    stars = extract_stars_from_csv()
    return render_template('rank_star.html', stars=stars)

if __name__ == '__main__':
    app.run(debug=True)
