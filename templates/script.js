document.addEventListener('DOMContentLoaded', function () {
    const emotionTable = document.getElementById('emotion-table');
    const emotionBody = document.getElementById('emotion-body');

    // Function to update the emotion table
    function updateEmotionTable(probabilities) {
        emotionBody.innerHTML = ''; // Clear existing rows

        const emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'];

        emotions.forEach((emotion, index) => {
            const probability = probabilities[index];
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${emotion}</td>
                <td>
                    <div class="bar-container">
                        <div class="bar">
                            <span style="width: ${probability * 100}%;"></span>
                        </div>
                        ${Math.round(probability * 100)}%
                    </div>
                </td>
            `;
            emotionBody.appendChild(row);
        });
    }

    // Function to update position and display table
    function updatePositionAndEmotions(x, y, probabilities) {
        emotionTable.style.left = `${x}px`;
        emotionTable.style.top = `${y}px`;
        emotionTable.classList.remove('hidden');
        updateEmotionTable(probabilities);
    }

    // Example function to simulate updating face position and emotion probabilities
    function simulateFaceDetection() {
        // Simulate some emotion probabilities
        const simulatedProbabilities = [0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.05];
        // Simulate face position
        const simulatedX = 100;
        const simulatedY = 100;
        updatePositionAndEmotions(simulatedX, simulatedY, simulatedProbabilities);
    }

    // Call simulate function every 1 second for demo purposes
    setInterval(simulateFaceDetection, 1000);

    // If you are getting real data from the server, use WebSocket or AJAX to update emotions
    /*
    const socket = new WebSocket('ws://your-server-url');
    socket.onmessage = function(event) {
        const data = JSON.parse(event.data);
        updatePositionAndEmotions(data.x, data.y, data.probabilities);
    };
    */
});
