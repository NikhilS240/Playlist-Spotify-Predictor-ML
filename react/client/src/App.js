import React, { useState } from 'react';

export default function Printer() {
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="app-container">
      <div className="app-header">
        <h1 className="app-title">Spotify Song Finder</h1>
        <p className="app-subtitle">Discover your next favorite track</p>
      </div>
      
      <MyForm setResult={setResult} setIsLoading={setIsLoading} isLoading={isLoading} />
      
      {isLoading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <p>Finding your perfect song... This may take a few minutes (P.S. There is a low chance it may not work with the specific playlist inputted. In this case, try another one) </p>
        </div>
      )}
      
      {result && !isLoading && (
        <div className="result-card">
          <h3 className="result-title">Your Recommendation</h3>
          <div className="result-info">
            <div className="result-item">
              <span className="result-label">Song:</span>
              <span className="result-value">{result.song}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Artist:</span>
              <span className="result-value">{result.artist}</span>
            </div>
            <div className="result-item">
              <span className="result-label">Album:</span>
              <span className="result-value">{result.album}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function MyForm({ setResult, setIsLoading, isLoading }) {
  function handleSubmit(e) {
    e.preventDefault();
    setIsLoading(true);
    setResult(''); // Clear previous results

    const form = e.target;
    const formData = new FormData(form);
    const formJson = Object.fromEntries(formData.entries());
    
    fetch('http://localhost:5000/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formJson),
    })
      .then(res => res.json())
      .then(data => {
        console.log("Raw Flask response:", data);
        console.log("Type of data:", typeof data);
        
        setResult(data);
        setIsLoading(false);
      })
      .catch(error => {
        setResult({ error: 'Error: ' + error.message });
        setIsLoading(false);
      });
  }

  return (
    <form className="recommendation-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label className="form-label">
          Spotify Playlist URL:
        </label>
        <input 
          className="form-input"
          name="myInput" 
          placeholder="https://open.spotify.com/playlist/..."
          required
          disabled={isLoading}
        />
      </div>
      <button 
        className="submit-button" 
        type="submit"
        disabled={isLoading}
      >
        {isLoading ? 'Processing...' : 'Get Recommendation'}
      </button>
    </form>
  );
}