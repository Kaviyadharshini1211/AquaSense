import React, { useState } from 'react';
import axios from 'axios';

function SearchBar() {
  const [city, setCity] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult('');
    setLoading(true);
    try {
      const backendResponse = await axios.post('http://localhost:8000/predict/city', {
        city: city
      });

      setResult(`Prediction: ${backendResponse.data.final_prediction.predicted_class}`);
    } catch (err) {
      setResult('Prediction failed: ' + (err?.response?.data?.detail || err.message));
    }
    setLoading(false);
  };

  return (
    <div className="search-container">
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Enter Indian city"
          value={city}
          onChange={(e) => setCity(e.target.value)}
          required
        />
        <button type="submit">Predict</button>
      </form>
      {loading && <p>⏳ Analyzing...</p>}
      {result && <p>{result}</p>}
    </div>
  );
}

export default SearchBar;
