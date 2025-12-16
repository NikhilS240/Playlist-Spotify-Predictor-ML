import React, { useState, useEffect } from 'react';

// HARDCODED - Simple and direct
const getAPIUrl = () => process.env.REACT_APP_API_URL;

export default function Printer() {
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const session = params.get('session');
    const error = params.get('error');
    
    if (session) {
      setSessionId(session);
      setIsAuthenticated(true);
      localStorage.setItem('spotify_session', session);
      window.history.replaceState({}, document.title, '/');
    } else if (error) {
      alert('Failed to connect to Spotify. Please try again.');
      window.history.replaceState({}, document.title, '/');
    } else {
      const savedSession = localStorage.getItem('spotify_session');
      if (savedSession) {
        setSessionId(savedSession);
        setIsAuthenticated(true);
      }
    }
  }, []);

  const handleSpotifyLogin = () => {
    const apiUrl = getAPIUrl();
    alert('About to redirect to: ' + apiUrl + '/login'); // Debug alert
    window.location.href = `${apiUrl}/login`;
  };

  return (
    <div className="app-container">
      <div className="app-header">
        <h1 className="app-title">Spotify Song Finder</h1>
        <p className="app-subtitle">Discover your next favorite track</p>
      </div>
      
      {!isAuthenticated ? (
        <div style={{textAlign: 'center', padding: '40px'}}>
          <p style={{marginBottom: '20px'}}>Connect your Spotify account to get started</p>
          <button className="submit-button" onClick={handleSpotifyLogin}>
            🎵 Connect Spotify
          </button>
        </div>
      ) : (
        <>
          <MyForm 
            setResult={setResult} 
            setIsLoading={setIsLoading} 
            isLoading={isLoading}
            sessionId={sessionId}
          />
          
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
        </>
      )}
    </div>
  );
}

function MyForm({ setResult, setIsLoading, isLoading, sessionId }) {
  function handleSubmit(e) {
    e.preventDefault();
    setIsLoading(true);
    setResult('');

    const form = e.target;
    const formData = new FormData(form);
    const formJson = Object.fromEntries(formData.entries());
    
    const apiUrl = getAPIUrl();
    
    fetch(`${apiUrl}/submit`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        ...formJson,
        sessionId: sessionId
      }),
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