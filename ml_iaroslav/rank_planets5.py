#!/usr/bin/env python
# coding: utf-8

# ## Ranking planets by their similarity using cosine similarity

# In[17]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import matplotlib.pyplot as plt
import shutil

# In[18]:


# Load dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df


# In[19]:


# Normalize the features for cosine similarity
def normalize_for_cosine_similarity(output_dir, df, features):
    new_df = df[features]
    try:
        new_df.to_csv(output_dir + '/planet_rank_features.csv', index=False)
    except Exception as e:
        print(f"Failed to save planet_rank_features.csv: {e}")

    X = new_df.values   
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_normalized


# In[20]:


def rank_by_similarity(X_normalized, planet_name, df):
    # Find the index of the selected planet
    target_index = df[df['Planet_Name'] == planet_name].index[0]
    
    # Compute cosine distance
    distances = cosine_distances(X_normalized[target_index].reshape(1, -1), X_normalized)
    
    # Sort by distance (similarity in reverse since lower distance means more similar)
    sorted_indices = np.argsort(distances[0])
    
    # Return sorted names with their similarity scores (1 - distance for similarity)
    # Exclude the planet itself from the ranking
    rankings = list(zip(df['Planet_Name'].iloc[sorted_indices], [1 - dist for dist in distances[0][sorted_indices]]))
    return [r for r in rankings if r[0] != planet_name]


# In[21]:

def plot_similarity_ranking(rankings, reference_planet, output_dir, plot_count=10):
    rankings_df = pd.DataFrame(rankings, columns=['Planet_Name', 'Similarity_Score'])
    top_10 = rankings_df.sort_values(by='Similarity_Score', ascending=False).head(plot_count)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.bar(top_10['Planet_Name'], top_10['Similarity_Score'])
    
    # Customize the plot
    plt.title(f"Top 10 Planets Most Similar to {reference_planet}")
    plt.xlabel("Planets")
    plt.ylabel("Similarity Score")
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()

    plt.savefig("planet_similarity_ranking.png", dpi=100)
    
    try:
        shutil.copy("planet_similarity_ranking.png", output_dir + '/planet_similarity_ranking.png')
    except Exception as e:
        print(f"Failed to copy planet_similarity_ranking: {e}")

    plt.show()


# In[22]:


def planet_rank_by_similarity(planet_name, file_path, output_dir, features):
    df = load_dataset(file_path)
    X_normalized = normalize_for_cosine_similarity(output_dir, df, features)
    ranks = rank_by_similarity(X_normalized, planet_name, df)
    return ranks


# In[ ]:

if __name__ == "__main__":
    planet_name = "CoRoT-31 b"
    ranks = planet_rank_by_similarity(planet_name, file_path="Resources/Cleaned Dataset.csv", output_dir="../Output/CSV_Files", features=["Planet_Mass_Earth", "Equilibrium_Temperature", "ra"])
    plot_similarity_ranking(rankings=ranks, reference_planet=planet_name, output_dir="../Output/Visualisations", plot_count=30)


