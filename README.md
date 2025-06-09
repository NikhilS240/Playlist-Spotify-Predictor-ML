
# Spotify Song Attribute Predictor

This project collects Spotify playlist data and extracts features like genres, artists, album types, popularity, length, and release year. It trains a simple neural network to predict likely attributes of the next songs you might enjoy. The model outputs predictions for album type, popularity, length, release year, and top genres and artists.

## Libraries Required

- `spotipy` — for Spotify API access  
- `torch` — for building and training the neural network
- and more
- Python built-ins: `collections`, `datetime`, `random`

## How to Run

1. Install dependencies (and Python 3 if you don't have it):
   ```bash
   pip install spotipy python-dotenv torch
   ```
2. Run the program:

   - On **Mac/Linux**:
     ```bash
     python ./dataCollection.py
     ```
   - On **Windows**:
     ```bash
     python3 ./dataCollection.py
     ```
