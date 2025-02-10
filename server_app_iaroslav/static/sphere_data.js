// Receive the star data from Flask
const stars = {{ stars | tojson }};
const x_coords = {{ x_coords | tojson }};
const y_coords = {{ y_coords | tojson }};
const z_coords = {{ z_coords | tojson }};

// Create the 3D scatter plot trace
const trace = {
    x: x_coords,
    y: y_coords,
    z: z_coords,
    mode: 'markers',
    type: 'scatter3d',
    text: stars,
    marker: {
        size: 5,
        color: 'rgba(255, 0, 0, 0.6)',
        line: { width: 2 }
    }
};

// Define layout for the celestial sphere
const layout = {
    title: 'Stars on the Celestial Sphere',
    scene: {
        xaxis: { showgrid: false, zeroline: false },
        yaxis: { showgrid: false, zeroline: false },
        zaxis: { showgrid: false, zeroline: false }
    },
    showlegend: false
};

// Plot the celestial sphere
Plotly.newPlot('chart', [trace], layout);
