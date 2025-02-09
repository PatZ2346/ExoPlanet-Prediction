#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import plotly.graph_objects as go
import pandas as pd


# In[2]:

def load_and_normalize_dataset(file_path, features):
    # Load dataset
    df = pd.read_csv(file_path)

    # Normalize the features for cosine similarity
    X = df[features].values
    X_normalized = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    return X_normalized, df

# In[3]:
def load_dataset_and_extract_star_names(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    host_star_list = df['Host_Star'].tolist()
    return host_star_list

# In[4]:

def rank_by_similarity(X_normalized, star_name, df):
    # Find the index of the selected star
    matching_stars = df[df['Host_Star'] == star_name]
    
    if matching_stars.empty:
        print(f"Error: Star '{star_name}' not found in the dataset.")
        return []
    
    target_index = matching_stars.index[0]
    
    # Compute cosine distance
    distances = cosine_distances(X_normalized[target_index].reshape(1, -1), X_normalized)
    
    # Sort by distance (similarity in reverse since lower distance means more similar)
    sorted_indices = np.argsort(distances[0])
    
    # Return sorted names with their similarity scores (1 - distance for similarity)
    # Exclude the star itself from the ranking
    rankings = list(zip(df['Host_Star'].iloc[sorted_indices], [1 - dist for dist in distances[0][sorted_indices]]))
    return [r for r in rankings if r[0] != star_name]


# In[5]:

def plotly_similarity_ranking_obj(rankings, reference_star):
    if not rankings:
        print("No data to plot.")
        return

    # Convert rankings to DataFrame for easier plotting with Plotly
    rankings_df = pd.DataFrame(rankings, columns=['Host_Star', 'Similarity_Score'])

    # Interactive Plot with Plotly
    fig = go.Figure(data=[
        go.Bar(
            x=rankings_df['Host_Star'],
            y=rankings_df['Similarity_Score'],
            text=rankings_df['Similarity_Score'],
            textposition='auto',
            hoverinfo='text+x',
            hovertext=[f"Similarity to {reference_star}: {score:.3f}" for score in rankings_df['Similarity_Score']]
        )
    ])

    # Update layout for better readability
    fig.update_layout(
        title=f"Similarity of Stars to {reference_star}",
        xaxis_title="Stars",
        yaxis_title="Similarity Score",
        xaxis_tickangle=-45,
        height=600
    )
    return fig

def plotly_similarity_ranking_html(rankings, reference_star):
    fig = plotly_similarity_ranking_obj(rankings, reference_star)
    chart_html = fig.to_html(full_html=False)  # This generates the Plotly chart in HTML format
    # print(chart_html)
    return chart_html

def plot_similarity_ranking(rankings, reference_star):
    fig = plotly_similarity_ranking_obj(rankings, reference_star)
    fig.write_html("star_similarity_ranking.html")
    # Show the plot
    fig.show()
    return fig

def extract_host_stars(file_path):
    df = pd.read_csv(file_path, usecols=['Host_Star', 'ra', 'dec'])
    df = df.drop_duplicates().dropna()
    return df.to_dict(orient='records')

# In[6]:


if __name__ == "__main__":
    # Example usage
    selected_star = "Kepler-107"  # Change this to any star in dataset
    # Rank and plot stars by similarity to the selected star
    X_normalized, df = load_and_normalize_dataset(file_path="../CSV_Files/Cleaned Dataset.csv", features=['Star_Temperature_K', 'Star_Radius_Solar', 'Star_Mass_Solar'])
    rankings = rank_by_similarity(X_normalized, selected_star, df)
    plot_similarity_ranking(rankings, selected_star)

