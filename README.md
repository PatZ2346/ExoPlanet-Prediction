# ExoPredict-Classifying-and-Predicting-Exoplanet-Characteristics

# Project Overview
   In this project, we aimed to analyze exoplanetary data to uncover insights into planetary classifications, habitability, and discovery methods using machine learning techniques. Our approach involved extracting, transforming, and visualizing data to address key research questions related to exoplanet clustering, classification, and relationships with their host stars.
   
# Technologies Used for ETL Process

  We leveraged various tools and frameworks to execute this project:

 Python: Primary programming language for data processing and machine learning.

 Pandas: Used for data manipulation and transformation.

 PySpark: Implemented large-scale data processing, clustering, and classification models.

 Matplotlib & Seaborn: Created visualizations to support data interpretation.

 Google Colab: Hosted and executed the project for collaborative development.
 
 # ETL Process (Extract, Transform, Load)

1. Data Extraction

We used a dataset containing exoplanet properties, orbital details, and discovery methods.

The dataset was read into a PySpark DataFrame from a CSV file.

2. Data Transformation

Cleaning: Handled missing values and standardized column names.

Feature Engineering: Selected relevant features for machine learning models.

Normalization & Encoding: Standardized numerical values and indexed categorical variables.

3. Data Loading & Analysis

The processed data was saved as structured CSV files for further analysis and visualization.

## Research Questions & Analysis

1. Can we identify groups of similar exoplanets?

Applied K-Means clustering on features like mass, radius, and temperature.

Created scatter plots to visualize clusters.

2. Can we classify planets into different categories (e.g., rocky, gas giants)?

Used Random Forest Classification to predict planet types based on key features.

Evaluated model accuracy and visualized class distributions.

3 & 4. Classifying planets by insolation, temperature, and mass

Analyzed planetary properties concerning stellar radiation and location.

Generated pair plots to visualize relationships between features.

5. How do planetary characteristics influence habitability?

Examined semi-major axis vs. temperature to determine habitability.

Saved results and plotted scatter diagrams.

6. How does a planet’s orbit impact its temperature and habitability?

Analyzed correlations between orbital distance and temperature.

Created line plots to showcase trends.

7. Is there a correlation between discovery facilities and methods used?

Grouped exoplanet discoveries by facility and detection method.

Visualized the distribution using bar charts.

8. How does host star distance influence planetary characteristics?

Studied the impact of host star distance on exoplanet temperature and mass.

Saved insights into structured CSV files.


# Resources
1. https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS&constraint=default_flag%3E0&constraint=disc_method%20like%20%27tran%27

