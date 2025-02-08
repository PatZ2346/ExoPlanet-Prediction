#!/usr/bin/env python
# coding: utf-8

# ## Ranking planets by their similarity using cosine similarity

# In[17]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import numpy as np
import plotly.graph_objects as go


# In[18]:


# Load dataset
def load_dataset(file_path="Resources/part-00000-3d57ee90-8dc9-4f89-97e6-768aa0ffce3c-c000.csv"):
    df = pd.read_csv(file_path)
    return df


# In[19]:


# Normalize the features for cosine similarity
def normalize_for_cosine_similarity(df, features = ['Insolation_Flux', 'Equilibrium_Temperature']):
    X = df[features].values
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


def plot_similarity_ranking(rankings, reference_planet):
    # Convert rankings to DataFrame for easier plotting with Plotly
    rankings_df = pd.DataFrame(rankings, columns=['Planet_Name', 'Similarity_Score'])

    # Interactive Plot with Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=rankings_df['Planet_Name'],
            y=rankings_df['Similarity_Score'],
            text=rankings_df['Similarity_Score'],
            textposition='auto',
            hoverinfo='text+x',
            hovertext=[f"Similarity to {reference_planet}: {score:.3f}" for score in rankings_df['Similarity_Score']]
        )
    ])

    # Update layout for better readability
    fig.update_layout(
        title=f"Similarity of Planets to {reference_planet}",
        xaxis_title="Planets",
        yaxis_title="Similarity Score",
        xaxis_tickangle=-45,
        height=600
    )

    # Show the plot
    fig.show()


# In[22]:


def planet_rank_by_similarity(planet_name, file_path="Resources/part-00000-3d57ee90-8dc9-4f89-97e6-768aa0ffce3c-c000.csv", features = ['Insolation_Flux', 'Equilibrium_Temperature']):
    df = load_dataset(file_path)
    X_normalized = normalize_for_cosine_similarity(df, features)
    ranks = rank_by_similarity(X_normalized, planet_name, df)
    return ranks


# In[ ]:


#planet_name = "CoRoT-31 b"
#ranks = planet_rank_by_similarity(planet_name)
#plot_similarity_ranking(ranks, planet_name)

