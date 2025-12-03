// frontend/src/App.jsx
import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [prediction, setPrediction] = useState("Loading...");

  useEffect(() => {
    const interval = setInterval(() => {
      fetch("http://localhost:8000/api/latest")
        .then((res) => res.json())
        .then((data) => {
          if (data.error) {
            setPrediction(`❌ Error: ${data.error}`);
          } else {
            setPrediction(
              `✅ Prediction: ${data.result} (Confidence: ${data.confidence})`
            );
          }
        })
        .catch((err) => setPrediction(`❌ Fetch error: ${err.message}`));
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="container">
      <h1>🖐️ Real-Time Gesture Prediction</h1>
      <div className="prediction-box">{prediction}</div>
    </div>
  );
}

export default App;
