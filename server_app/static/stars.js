// stars.js

// Ensure the data passed from Flask is available
console.log("Stars:", stars);
console.log("X Coordinates:", x_coords);
console.log("Y Coordinates:", y_coords);
console.log("Z Coordinates:", z_coords);

// Set the radius of the celestial sphere (you can adjust this value)
// Change this value to scale the sphere as needed
const radius = 2; 

// Normalize coordinates to lie on the surface of the sphere (if they aren't already)
const normalizeCoordinates = (x, y, z, radius) => {
    const length = Math.sqrt(x * x + y * y + z * z);
    return {
        x: x / length * radius,
        y: y / length * radius,
        z: z / length * radius
    };
};

// Calculate distance from the origin (0, 0, 0) for each star
const calculateDistance = (x, y, z) => {
    return Math.sqrt(x * x + y * y + z * z);
};

// Normalize all star coordinates
const normalizedCoordinates = x_coords.map((x, index) => {
    const y = y_coords[index];
    const z = z_coords[index];
    return normalizeCoordinates(x, y, z, radius);
});

// Extract the normalized coordinates back into separate arrays for Plotly
const normalizedX = normalizedCoordinates.map(coord => coord.x);
const normalizedY = normalizedCoordinates.map(coord => coord.y);
const normalizedZ = normalizedCoordinates.map(coord => coord.z);

// Calculate the distance for each star
const distances = normalizedCoordinates.map((coord) => calculateDistance(coord.x, coord.y, coord.z));

// Create a color scale based on distance
// Map distance range to color scale
const colorScale = d3.scaleSequential(d3.interpolateViridis)
    .domain([Math.min(...distances), Math.max(...distances)]); 

// Map distances to colors
const starColors = distances.map(distance => colorScale(distance));

// Create the 3D scatter plot trace
const trace = {
    x: normalizedX,
    y: normalizedY,
    z: normalizedZ,
    mode: 'markers',
    type: 'scatter3d',
    text: stars,
    marker: {
        size: 5,
        color: starColors,
        colorscale: 'Viridis',
        colorbar: {
            title: 'Distance'
        },
        line: { width: 2 }
    }
};

// Define layout for the celestial sphere with grid, axis, and responsiveness
const layout = {
    title: 'Stars on the Celestial Sphere',
    scene: {
        xaxis: {
            showgrid: true,
            zeroline: true,
            title: 'X Axis'
        },
        yaxis: {
            showgrid: true,
            zeroline: true,
            title: 'Y Axis'
        },
        zaxis: {
            showgrid: true,
            zeroline: true,
            title: 'Z Axis'
        }
    },
    showlegend: false,
    responsive: true // Make the plot responsive to window size
};

Plotly.newPlot('chart', [trace], layout);

window.onresize = function() {
    Plotly.Plots.resize(document.getElementById('chart'));
};
