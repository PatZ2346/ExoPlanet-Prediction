function rankStar() {
    let selectedStar = document.getElementById("starSelect").value;
    if (selectedStar) {
        fetch(`/generate-chart?star=${encodeURIComponent(selectedStar)}`)
            .then(response => response.json())  // Expect JSON response
            .then(data => {
                if (data.chart_filename) {
                    window.location.href = `/view-chart/${encodeURIComponent(data.chart_filename)}`;
                } else {
                    console.error("No chart filename returned");
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    }
}
