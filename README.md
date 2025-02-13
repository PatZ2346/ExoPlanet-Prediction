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

#### We have chosen following questions for our data analysis:
Planet’s habitability
Stellar orbital Impact on planet’s temperature
Categorisation of planets (Gas Giant or Rocky surface)
Which facility discovered most exoplanets
What methods are more acceptable than others to discover the exoplanets and why?
Comparing stars and exoplanets by how far their parameters are from each other
Trying to predict number of exoplanets per star



## 1. How do planetary characteristics influence the likelihood of being in the habitable zone? 

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




## 2. How does a planet’s orbit impact its temperature and potential habitability?

Answer: The orbital eccentricity and semi-major axis influence temperature variations. We will analyze how eccentric orbits lead to temperature fluctuations and impact habitability.

   Highly eccentric orbits (e > 0.3) cause significant temperature fluctuations during different phases of orbit. We will compare eccentric vs. circular orbits to see how temperature variability impacts habitability.

Analysis: 
    Highly Eccentric Orbits (e > 0.3) show greater temperature variations, meaning these planets experience extreme seasonal changes.
   Circular/Stable Orbits (e ≤ 0.3) have more stable temperatures, making them better candidates for habitability.
   If a planet has high eccentricity, it may enter and exit the habitable zone multiple times during an orbit, making sustained habitability unlikely.

## Which exoplanet discovery method has been used the most, and which has been used the least? What does this indicate about the effectiveness and prevalence of different detection techniques? (Amrit)
 Transit is the most dominant method for discovering exoplanets, while Transit Timing Variations (TTV) is the least used.

- Transit (1102 discoveries) → Most widely used method, likely due to missions like Kepler and TESS, which monitor star brightness dips when planets pass in front.
- Transit Timing Variations (1 discovery) → Rarely used, likely because it requires detecting minute gravitational influences on already-known planets.

This analysis highlights the dominance of large-scale transit surveys and the rarity of more specialized detection techniques like TTV

## How does the distance of a planet host from its exoplanets influence the exoplanets' temperature, luminosity?

The analysis compares the exoplanets OGLE-TR-182 b (farthest from its star at 2501.75 light-years) and HD 219134 b (closest at 6.53 light-years).

It suggests that both exoplanets despite being farthest and closest to their host, are both depict high temperature where as the luminosity difference is apparently high in the exoplanet closest to its star at 14.737.

However, analyzing the visualizations reveals a clear inverse relationship between an exoplanet’s distance from its host star and its temperature. Planets closer to their stars tend to be hotter, while those farther away are cooler, following established astrophysical principles. This study highlights extreme planetary diversity, where distant gas giants and nearby rocky planets provide unique opportunities for exoplanetary science.

Can we identify groups of similar exoplanets?

The K-Means algorithm clusters exoplanets based on mass and temperature into three groups (0, 1, 2). Cluster 2 contains most planets, including Kepler-138 c and HD 136352 c, with moderate mass (2.3–11.24) and temperatures (410–784K). Cluster 0 includes massive, hotter planets like TOI-3819 b (230.11–371.85 mass, 1028–1890K). This analysis highlights mass and temperature as key factors in exoplanet classification.


Techniques implemented for Optimization: 1. Increased Network Depth and Width2. Batch Normalization3. Dropout Regularization4. Early Stopping5. Learning Rate Reduction6. Extended Training Epochs7. Larger Batch Size8. Performance Tracking


## Key Findings
-   Total 634 Planets were found habitable
-   We are surrounded by more Gas Giants than Rocky Planets
-   Transit method is widely used to discover the new exoplanets
-   Planets closer to their host are highly luminous and hot
-   The mass of the star significantly impacts on the number of exoplanets


## Recommendations
Higher resolution datasets
Refining models
Incorporating additional planetary characteristics 

## Conclusion
The ExoPredict project successfully analyzed exoplanetary data to classify and predict exoplanet characteristics using machine learning and clustering algorithms. By leveraging NASA's exoplanet dataset, our team explored planetary properties, habitability, and discovery trends.






# Resources
1. https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS&constraint=default_flag%3E0&constraint=disc_method%20like%20%27tran%27

