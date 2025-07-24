import os
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import math
import random

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrackFeatures:
    """Data class to hold track features"""
    genres: List[int]
    artists: List[int]
    album_type: int
    popularity: float
    length: float
    years_old: float
    track_name: str = ""

class SpotifyRecommender:
    """Main class for Spotify song recommendation system"""
    
    def __init__(self, database_file: str = "database.txt"):
        self.database_file = Path(database_file)
        self.sp = self._setup_spotify_client()
        self.artist_frequency = Counter()
        
    def _setup_spotify_client(self) -> spotipy.Spotify:
        """Setup Spotify API client with credentials"""
        load_dotenv()
        client_id = os.getenv("client_id")
        client_secret = os.getenv("client_secret")
        
        if not client_id or not client_secret:
            raise ValueError("Spotify credentials not found in environment variables")
        
        return spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))
    
    def extract_playlist_features(self, playlist_id: str) -> Tuple[List[TrackFeatures], List[str], List[str]]:
        """Extract features from a Spotify playlist"""
        logger.info(f"Processing playlist: {playlist_id}")
        
        offset = 0
        limit = 100
        track_features = []
        all_genres = []
        all_artists = []
        track_names = []
        
        while True:
            try:
                playlist_data = self.sp.playlist_items(playlist_id, offset=offset, limit=limit)
                items = playlist_data['items']
                
                if not items:
                    break
                
                for item in items:
                    if not item.get('track') or not item['track'].get('album'):
                        continue
                    
                    features = self._extract_track_features(item['track'], all_genres, all_artists)
                    if features:
                        track_features.append(features)
                        track_names.append(item['track']['name'])
                
                offset += limit
                
            except SpotifyException as e:
                logger.error(f"Spotify API error: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        
        logger.info(f"Processed {len(track_features)} tracks")
        return track_features, all_genres, all_artists, track_names
    
    def _extract_track_features(self, track: Dict, all_genres: List[str], all_artists: List[str]) -> Optional[TrackFeatures]:
        """Extract features from a single track"""
        try:
            album = track['album']
            length = track['duration_ms']
            release_date = album['release_date']
            album_type = album['album_type']
            popularity = track['popularity']
            
            # Process release year
            year_str = release_date.split("-")[0]
            release_year = int(year_str)
            current_year = datetime.now().year
            years_old = (current_year - release_year) / 10
            
            # Normalize features
            popularity_norm = popularity / 100
            length_norm = round(length / 600000, 3)
            album_type_binary = 1 if album_type == 'album' else 0
            
            # Extract artist and genre information
            genre_indices = []
            artist_indices = []
            
            for artist in album['artists']:
                try:
                    artist_data = self.sp.artist(artist['id'])
                    artist_name = artist_data['name']
                    artist_genres = artist_data['genres']
                    
                    # Update frequency counter
                    self.artist_frequency[artist_name] += 1
                    
                    # Add to master lists if not present
                    if artist_name not in all_artists:
                        all_artists.append(artist_name)
                    artist_indices.append(all_artists.index(artist_name))
                    
                    # Process genres
                    for genre in artist_genres:
                        if genre not in all_genres:
                            all_genres.append(genre)
                        if genre not in [all_genres[i] for i in genre_indices]:
                            genre_indices.append(all_genres.index(genre))
                            
                except SpotifyException:
                    continue
            
            # Fallback for missing data
            if not genre_indices:
                genre_indices = [9999]  # Unknown genre marker
            
            return TrackFeatures(
                genres=genre_indices,
                artists=artist_indices,
                album_type=album_type_binary,
                popularity=popularity_norm,
                length=length_norm,
                years_old=years_old,
                track_name=track.get('name', '')
            )
            
        except Exception as e:
            logger.warning(f"Error processing track: {e}")
            return None

class MusicDataset(Dataset):
    """PyTorch Dataset for music recommendation"""
    
    def __init__(self, track_features: List[TrackFeatures], num_genres: int, num_artists: int):
        self.track_features = track_features
        self.num_genres = num_genres
        self.num_artists = num_artists
        
    def __len__(self):
        return len(self.track_features)
    
    def __getitem__(self, idx):
        features = self.track_features[idx]
        
        # Create multi-hot vectors
        genre_vector = self._create_multi_hot(features.genres, self.num_genres)
        artist_vector = self._create_multi_hot(features.artists, self.num_artists)
        
        return {
            'genre': torch.tensor(genre_vector, dtype=torch.float32),
            'artist': torch.tensor(artist_vector, dtype=torch.float32),
            'album_type': torch.tensor(features.album_type, dtype=torch.long),
            'popularity': torch.tensor(features.popularity, dtype=torch.float32),
            'length': torch.tensor(features.length, dtype=torch.float32),
            'years_old': torch.tensor(features.years_old, dtype=torch.float32)
        }
    
    def _create_multi_hot(self, indices: List[int], num_classes: int) -> List[int]:
        """Create multi-hot encoding from indices"""
        multi_hot = [0] * num_classes
        for idx in indices:
            if 0 <= idx < num_classes:
                multi_hot[idx] = 1
        return multi_hot

class MusicRecommendationModel(nn.Module):
    """Neural network model for music recommendation"""
    
    def __init__(self, num_genres: int, num_artists: int, 
                 genre_embedding_dim: int = 16, artist_embedding_dim: int = 16,
                 hidden_dim: int = 64):
        super().__init__()
        
        self.genre_embedding = nn.Linear(num_genres, genre_embedding_dim)
        self.artist_embedding = nn.Linear(num_artists, artist_embedding_dim)
        
        input_dim = genre_embedding_dim + artist_embedding_dim + 4  # +4 for other features
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 6)  # 3 classification + 3 regression
        )
        
    def forward(self, genre, artist, album_type, popularity, length, years_old):
        # Create embeddings
        genre_emb = self.genre_embedding(genre)
        artist_emb = self.artist_embedding(artist)
        
        # Combine all features
        other_features = torch.stack([
            album_type.float(), popularity, length, years_old
        ], dim=1)
        
        combined = torch.cat([genre_emb, artist_emb, other_features], dim=1)
        
        # Get predictions
        output = self.classifier(combined)
        
        return {
            'classification': output[:, :3],  # Album type classification
            'regression': output[:, 3:]       # Popularity, length, years_old
        }

class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, model: MusicRecommendationModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        
    def train(self, train_loader: DataLoader, num_epochs: int = 20) -> List[float]:
        """Train the model"""
        self.model.train()
        loss_history = []
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            for batch in train_loader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(**batch)
                
                # Calculate losses
                class_loss = self.classification_loss(
                    predictions['classification'], 
                    batch['album_type']
                )
                
                reg_targets = torch.stack([
                    batch['popularity'],
                    batch['length'],
                    batch['years_old']
                ], dim=1)
                
                reg_loss = self.regression_loss(predictions['regression'], reg_targets)
                
                # Combined loss
                total_batch_loss = class_loss + reg_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += total_batch_loss.item()
            
            avg_loss = total_loss / len(train_loader)
            loss_history.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return loss_history
    
    def predict_preferences(self, data_loader: DataLoader) -> Dict:
        """Generate predictions for the entire dataset"""
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predictions = self.model(**batch)
                
                # Combine predictions
                combined_pred = torch.cat([
                    predictions['classification'],
                    predictions['regression']
                ], dim=1)
                
                all_predictions.append(combined_pred)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        mean_predictions = torch.mean(all_predictions, dim=0)
        
        return {
            'album_type': torch.argmax(mean_predictions[:3]).item(),
            'popularity': mean_predictions[3].item() * 100,
            'length': abs(mean_predictions[4].item() * 600000),
            'years_old': abs(mean_predictions[5].item() * 10)
        }

class RecommendationEngine:
    """Handles the recommendation logic"""
    
    def __init__(self, recommender: SpotifyRecommender):
        self.recommender = recommender
    
    def get_top_preferences(self, dataset: MusicDataset, top_k: int = 3) -> Tuple[List[str], List[str]]:
        """Get top genres and artists from the dataset"""
        # Calculate average weights
        all_genre_vectors = torch.stack([dataset[i]['genre'] for i in range(len(dataset))])
        all_artist_vectors = torch.stack([dataset[i]['artist'] for i in range(len(dataset))])
        
        avg_genre_weights = torch.mean(all_genre_vectors, dim=0)
        avg_artist_weights = torch.mean(all_artist_vectors, dim=0)
        
        # Get top indices
        top_genre_indices = torch.topk(avg_genre_weights, min(top_k, len(avg_genre_weights))).indices.tolist()
        top_artist_indices = torch.topk(avg_artist_weights, min(top_k, len(avg_artist_weights))).indices.tolist()
        
        return top_genre_indices, top_artist_indices
    
    def find_recommendation(self, predictions: Dict, genres: List[str], artists: List[str],
                          top_genre_indices: List[int], top_artist_indices: List[int],
                          track_names: List[str]) -> Optional[Dict]:
        """Find a song recommendation based on predictions"""
        
        # Get predicted values
        predicted_genres = [genres[i] for i in top_genre_indices if i < len(genres)]
        predicted_artists = [artists[i] for i in top_artist_indices if i < len(artists)]
        
        if not predicted_artists:
            logger.warning("No predicted artists found")
            return None
        
        logger.info("\n--- Predicted Song Attributes ---")
        logger.info(f"Album Type: {'album' if predictions['album_type'] == 1 else 'single'}")
        logger.info(f"Popularity: {predictions['popularity']:.1f}")
        logger.info(f"Length: {predictions['length']:.0f}ms")
        logger.info(f"Predicted Genres: {predicted_genres}")
        logger.info(f"Predicted Artists: {predicted_artists}")
        
        # Select artist using weighted random selection
        selected_artist = self._select_artist_weighted(predicted_artists, predicted_genres)
        if not selected_artist:
            return None
        
        # Find songs by the selected artist
        candidate_songs = self._find_candidate_songs(
            selected_artist, predictions, track_names
        )
        
        if not candidate_songs:
            logger.warning("No candidate songs found")
            return None
        
        # Select best matching song
        recommendation = self._select_best_song(candidate_songs, predictions)
        
        return recommendation
    
    def _select_artist_weighted(self, predicted_artists: List[str], predicted_genres: List[str]) -> Optional[str]:
        """Select an artist using weighted random selection based on genre matching"""
        matches = []
        
        for artist in predicted_artists:
            try:
                results = self.recommender.sp.search(q=f'artist:{artist}', type='artist', limit=1)
                if not results['artists']['items']:
                    matches.append(0)
                    continue
                
                artist_data = results['artists']['items'][0]
                artist_genres = artist_data['genres']
                
                # Count genre matches
                match_count = sum(1 for pg in predicted_genres for ag in artist_genres if pg == ag)
                
                # Apply bias for frequent artists
                most_common_artist = self.recommender.artist_frequency.most_common(1)[0][0]
                if artist == most_common_artist:
                    total_tracks = sum(self.recommender.artist_frequency.values())
                    artist_freq = self.recommender.artist_frequency[artist]
                    bias = 1.01 + (1.49 * (artist_freq / total_tracks))  # Bias between 1.01 and 2.5
                    match_count *= bias
                
                matches.append(match_count)
                
            except SpotifyException:
                matches.append(0)
        
        # Weighted random selection
        total_weight = sum(matches)
        if total_weight == 0:
            return predicted_artists[0] if predicted_artists else None
        
        rand_val = random.random() * total_weight
        cumulative = 0
        
        for i, weight in enumerate(matches):
            cumulative += weight
            if rand_val <= cumulative:
                return predicted_artists[i]
        
        return predicted_artists[0]
    
    def _find_candidate_songs(self, artist_name: str, predictions: Dict, existing_tracks: List[str]) -> List[Dict]:
        """Find candidate songs by the selected artist"""
        try:
            # Get artist ID
            results = self.recommender.sp.search(q=f'artist:{artist_name}', type='artist', limit=1)
            if not results['artists']['items']:
                return []
            
            artist_id = results['artists']['items'][0]['id']
            
            # Determine album type preference
            album_type = 'album' if random.random() < (0.9 if predictions['album_type'] == 1 else 0.4) else 'single'
            
            # Get albums
            albums = self._get_artist_albums(artist_id, album_type)
            if not albums:
                return []
            
            # Filter albums by release year
            target_year = datetime.now().year - int(predictions['years_old'])
            filtered_albums = self._filter_albums_by_year(albums, target_year)
            
            if not filtered_albums:
                return []
            
            # Get songs from albums
            candidate_songs = self._extract_songs_from_albums(filtered_albums, album_type, existing_tracks)
            
            return candidate_songs
            
        except Exception as e:
            logger.error(f"Error finding candidate songs: {e}")
            return []
    
    def _get_artist_albums(self, artist_id: str, album_type: str) -> List[Dict]:
        """Get all albums of specified type for an artist"""
        albums = []
        offset = 0
        
        while True:
            try:
                batch = self.recommender.sp.artist_albums(
                    artist_id, album_type=album_type, limit=50, offset=offset
                )
                items = batch['items']
                albums.extend(items)
                
                if len(items) < 50:
                    break
                offset += 50
                
            except SpotifyException:
                break
        
        return albums
    
    def _filter_albums_by_year(self, albums: List[Dict], target_year: int, max_expansion: int = 5) -> List[Dict]:
        """Filter albums by release year with expanding search window"""
        for expansion in range(1, max_expansion + 1):
            window = 2 * expansion
            lower_bound = target_year - window
            upper_bound = target_year + window
            
            filtered = []
            for album in albums:
                try:
                    release_year = int(album['release_date'].split('-')[0])
                    if lower_bound <= release_year <= upper_bound:
                        filtered.append(album)
                except (ValueError, IndexError):
                    continue
            
            if filtered:
                return filtered
        
        return albums  # Return all if no matches in any window
    
    def _extract_songs_from_albums(self, albums: List[Dict], album_type: str, existing_tracks: List[str]) -> List[Dict]:
        """Extract individual songs from albums"""
        songs = []
        
        if album_type == 'single':
            # For singles, just use the album name as the track name
            for album in albums:
                songs.append({
                    'name': album['name'],
                    'album_name': album['name']
                })
        else:
            # For albums, get all tracks
            for album in albums:
                try:
                    album_data = self.recommender.sp.search(
                        q=f"album:{album['name']}", type='album', limit=1
                    )['albums']['items']
                    
                    if album_data:
                        album_id = album_data[0]['id']
                        tracks = self.recommender.sp.album_tracks(album_id)['items']
                        
                        for track in tracks:
                            songs.append({
                                'name': track['name'],
                                'album_name': album['name']
                            })
                            
                except SpotifyException:
                    continue
        
        # Filter out existing tracks
        filtered_songs = [
            song for song in songs 
            if song['name'].strip().lower() not in [track.strip().lower() for track in existing_tracks]
        ]
        
        return filtered_songs
    
    def _select_best_song(self, candidate_songs: List[Dict], predictions: Dict) -> Optional[Dict]:
        """Select the best matching song from candidates"""
        song_scores = []
        
        predicted_length = predictions['length'] / 1000  # Convert to seconds
        predicted_popularity = predictions['popularity']
        
        for song in candidate_songs:
            try:
                # Search for the track to get its features
                search_results = self.recommender.sp.search(
                    q=f'track:{song["name"]}', type='track', limit=1
                )
                
                if not search_results['tracks']['items']:
                    continue
                
                track = search_results['tracks']['items'][0]
                song_length = track['duration_ms'] / 1000
                song_popularity = track['popularity']
                
                # Calculate distance (lower is better)
                distance = math.sqrt(
                    (predicted_length - song_length) ** 2 +
                    (predicted_popularity - song_popularity) ** 2
                )
                
                song_scores.append((distance, song, track))
                
            except SpotifyException:
                continue
        
        if not song_scores:
            return None
        
        # Sort by distance (ascending) and select from top matches with some randomness
        song_scores.sort(key=lambda x: x[0])
        
        # Select from top 3 matches randomly to add variety
        top_matches = song_scores[:min(3, len(song_scores))]
        selected = random.choice(top_matches)
        
        distance, song_info, track_data = selected
        
        # Get Spotify URL
        track_id = track_data['id']
        spotify_url = f"https://open.spotify.com/track/{track_id}"
        
        return {
            'name': song_info['name'],
            'artist': track_data['artists'][0]['name'],
            'album': song_info['album_name'],
            'url': spotify_url,
            'distance': distance
        }

class SpotifyRecommenderApp:
    """Main application class"""
    
    def __init__(self):
        self.recommender = SpotifyRecommender()
        self.recommendation_engine = RecommendationEngine(self.recommender)
        
    def run(self):
        """Main application loop"""
        print("=" * 50)
        print("🎵 SPOTIFY SONG RECOMMENDER 🎵")
        print("=" * 50)
        
        while True:
            self._show_menu()
            choice = input("\nWhat would you like to do? ")
            
            if not choice.isdigit():
                print("❌ Please enter a valid number.")
                continue
            
            choice = int(choice)
            
            if choice == 1:
                self._recommend_from_link()
            elif choice == 2:
                self._recommend_from_saved()
            elif choice == 3:
                self._save_playlist()
            elif choice == 4:
                self._clear_database()
            elif choice == 5:
                print("👋 Thank you for using Spotify Song Recommender!")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def _show_menu(self):
        """Display the main menu"""
        print("\n" + "─" * 50)
        print("MENU OPTIONS:")
        print("1. 🎧 Get recommendation from Spotify playlist link")
        print("2. 💾 Get recommendation from saved playlist")
        print("3. 📁 Save playlist for later use")
        print("4. 🗑️  Clear saved playlists")
        print("5. 🚪 Exit")
        print("─" * 50)
    
    def _recommend_from_link(self):
        """Get recommendation from a playlist link"""
        playlist_link = input("🔗 Enter Spotify playlist link: ").strip()
        
        if not playlist_link:
            print("❌ No link provided.")
            return
        
        try:
            recommendation = self._generate_recommendation(playlist_link)
            if recommendation:
                self._display_recommendation(recommendation)
            else:
                print("❌ Could not generate recommendation. Please try again.")
                
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            print("❌ An error occurred. Please check the playlist link and try again.")
    
    def _recommend_from_saved(self):
        """Get recommendation from a saved playlist"""
        if not self.recommender.database_file.exists():
            print("❌ No saved playlists found.")
            return
        
        with open(self.recommender.database_file, 'r') as f:
            playlists = [line.strip() for line in f.readlines() if line.strip()]
        
        if not playlists:
            print("❌ No saved playlists found.")
            return
        
        print("\n📋 SAVED PLAYLISTS:")
        for i, playlist_id in enumerate(playlists):
            try:
                playlist_info = self.recommender.sp.playlist(playlist_id)
                print(f"{i}: {playlist_info['name']}")
            except SpotifyException:
                print(f"{i}: Invalid playlist (ID: {playlist_id[:20]}...)")
        
        try:
            selection = int(input(f"\nSelect playlist (0-{len(playlists)-1}): "))
            if 0 <= selection < len(playlists):
                recommendation = self._generate_recommendation(playlists[selection])
                if recommendation:
                    self._display_recommendation(recommendation)
                else:
                    print("❌ Could not generate recommendation.")
            else:
                print("❌ Invalid selection.")
        except ValueError:
            print("❌ Please enter a valid number.")
    
    def _save_playlist(self):
        """Save a playlist for later use"""
        playlist_link = input("🔗 Enter Spotify playlist link to save: ").strip()
        
        if not playlist_link:
            print("❌ No link provided.")
            return
        
        try:
            # Validate playlist
            playlist_info = self.recommender.sp.playlist(playlist_link)
            
            # Save to database
            with open(self.recommender.database_file, 'a') as f:
                f.write(f"{playlist_link}\n")
            
            print(f"✅ Playlist '{playlist_info['name']}' saved successfully!")
            
        except SpotifyException:
            print("❌ Invalid playlist link. Not saved.")
    
    def _clear_database(self):
        """Clear all saved playlists"""
        if self.recommender.database_file.exists():
            self.recommender.database_file.unlink()
        print("✅ All saved playlists cleared!")
    
    def _generate_recommendation(self, playlist_id: str) -> Optional[Dict]:
        """Generate a song recommendation from a playlist"""
        try:
            # Extract features
            print("🔄 Analyzing playlist...")
            track_features, genres, artists, track_names = self.recommender.extract_playlist_features(playlist_id)
            
            if len(track_features) < 5:
                print("❌ Playlist too small for meaningful recommendation (need at least 5 tracks).")
                return None
            
            # Create dataset
            dataset = MusicDataset(track_features, len(genres), len(artists))
            data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            # Create and train model
            print("🤖 Training recommendation model...")
            model = MusicRecommendationModel(len(genres), len(artists))
            trainer = ModelTrainer(model)
            
            # Train model
            trainer.train(data_loader, num_epochs=15)
            
            # Generate predictions
            print("🎯 Generating predictions...")
            predictions = trainer.predict_preferences(data_loader)
            
            # Get top preferences
            top_genre_indices, top_artist_indices = self.recommendation_engine.get_top_preferences(dataset)
            
            # Find recommendation
            print("🔍 Finding recommendation...")
            recommendation = self.recommendation_engine.find_recommendation(
                predictions, genres, artists, top_genre_indices, top_artist_indices, track_names
            )
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in recommendation generation: {e}")
            return None
    
    def _display_recommendation(self, recommendation: Dict):
        """Display the final recommendation"""
        print("\n" + "🎵" * 25)
        print("✨ RECOMMENDATION ✨")
        print("🎵" * 25)
        print(f"🎧 Song: {recommendation['name']}")
        print(f"👤 Artist: {recommendation['artist']}")
        print(f"💿 Album: {recommendation['album']}")
        print(f"🔗 Listen: {recommendation['url']}")
        print(f"📊 Match Score: {recommendation['distance']:.2f}")
        print("🎵" * 25)

def main():
    """Entry point for the application"""
    try:
        app = SpotifyRecommenderApp()
        app.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print("❌ An unexpected error occurred. Please check your setup and try again.")

if __name__ == "__main__":
    main()