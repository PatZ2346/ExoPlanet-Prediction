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

## 1. How do planetary characteristics influence the likelihood of being in the habitable zone? (Mandeep)

Answer: The habitable zone (HZ) depends on the semi-major axis (AU) and star temperature (K). We'll classify planets as potentially habitable based on their distance from the star (semi-major axis) and the equilibrium temperature.

    We'll consider a planet inside the habitable zone if: Semi-major axis (AU) falls within the range for a Sun-like star (0.38 - 1.5 AU).

    Equilibrium temperature falls between 200K and 320K.
    Analysis: 
      Only 7 out of 1130 planets (~0.62%) fall into the potentially habitable category.
         Most planets do not meet the criteria for being in the habitable zone.
Key Influencing Factors:
        Semi-Major Axis (AU): Planets too close or too far from their stars receive too much or too little radiation, making them unsuitable for life.
       Equilibrium Temperature (K): Planets with extreme temperatures (very high or low) are unlikely to support liquid water.
       Star Temperature (K): Cooler stars (red dwarfs) have closer habitable zones, while hotter stars require planets to be farther out.

## 2. How does a planet’s orbit impact its temperature and potential habitability?(Mandeep)

Answer: The orbital eccentricity and semi-major axis influence temperature variations. We will analyze how eccentric orbits lead to temperature fluctuations and impact habitability.

   Highly eccentric orbits (e > 0.3) cause significant temperature fluctuations during different phases of orbit. We will compare eccentric vs. circular orbits to see how temperature variability impacts habitability.

Analysis: 
    Highly Eccentric Orbits (e > 0.3) show greater temperature variations, meaning these planets experience extreme seasonal changes.
   Circular/Stable Orbits (e ≤ 0.3) have more stable temperatures, making them better candidates for habitability.
   If a planet has high eccentricity, it may enter and exit the habitable zone multiple times during an orbit, making sustained habitability unlikely.


# Resources
1. https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS&constraint=default_flag%3E0&constraint=disc_method%20like%20%27tran%27

