import os
import json
import logging
import random
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter, defaultdict

from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrackData:
    """Simple track data structure using only available Spotify data"""
    track_id: str
    name: str
    artist: str
    album: str
    popularity: int
    duration_ms: int
    release_year: int
    genres: List[str]
    artist_popularity: int = 50
    explicit: bool = False

@dataclass
class UserTaste:
    """User's music taste profile using available data"""
    favorite_genres: Dict[str, float]
    favorite_artists: Dict[str, float]
    preferences: Dict[str, float]  # popularity, duration, etc.
    diversity_score: float

class SimpleSpotifyClient:
    """Simplified Spotify client with timeout handling"""
    
    def __init__(self):
        self.sp = self._setup_spotify()
        self.cache = {}
        self.request_delay = 0.1  # Small delay between requests
        
    def _setup_spotify(self):
        """Setup Spotify client with longer timeout"""
        load_dotenv()
        client_id = os.getenv("SPOTIFY_CLIENT_ID") or os.getenv("client_id")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET") or os.getenv("client_secret")
        
        if not client_id or not client_secret:
            raise ValueError("Missing Spotify credentials. Add SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET to .env file")
        
        # Create client with longer timeout
        client = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id, 
                client_secret=client_secret
            ),
            requests_timeout=30,  # Increase timeout to 30 seconds
            retries=3  # Add retries
        )
        
        return client
    
    def _make_request_with_retry(self, func, *args, max_retries=3, **kwargs):
        """Make Spotify API request with retry logic"""
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                    logger.info(f"Retrying request in {delay} seconds... (attempt {attempt + 1})")
                    time.sleep(delay)
                
                result = func(*args, **kwargs)
                
                # Add small delay between successful requests
                time.sleep(self.request_delay)
                return result
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Request attempt {attempt + 1} failed: {e}")
        
        return None
    
    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """Get all tracks from playlist with timeout handling"""
        tracks = []
        offset = 0
        max_tracks = 500  # Limit to prevent very long requests
        
        logger.info(f"Fetching tracks from playlist {playlist_id}")
        
        while len(tracks) < max_tracks:
            try:
                logger.info(f"Fetching tracks {offset}-{offset+99}")
                
                results = self._make_request_with_retry(
                    self.sp.playlist_tracks,
                    playlist_id, 
                    offset=offset, 
                    limit=100
                )
                
                if not results or not results['items']:
                    break
                
                batch_tracks = []
                for item in results['items']:
                    if item and item.get('track') and item['track'].get('id'):
                        batch_tracks.append(item['track'])
                
                tracks.extend(batch_tracks)
                logger.info(f"Retrieved {len(batch_tracks)} tracks (total: {len(tracks)})")
                
                offset += 100
                
                # Stop if we got less than requested (end of playlist)
                if len(results['items']) < 100:
                    break
                
            except SpotifyException as e:
                logger.error(f"Spotify API error at offset {offset}: {e}")
                if "timeout" in str(e).lower() or "connection" in str(e).lower():
                    logger.info("Continuing with tracks retrieved so far...")
                    break
                else:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error at offset {offset}: {e}")
                break
        
        logger.info(f"Successfully retrieved {len(tracks)} tracks total")
        return tracks
    
    def get_artist_info(self, artist_id: str) -> Optional[Dict]:
        """Get artist information with caching and timeout handling"""
        if artist_id in self.cache:
            return self.cache[artist_id]
        
        try:
            artist = self._make_request_with_retry(self.sp.artist, artist_id)
            if artist:
                self.cache[artist_id] = artist
            return artist
        except Exception as e:
            logger.error(f"Error getting artist info for {artist_id}: {e}")
            return None
    
    def search_similar_tracks(self, seed_artists: List[str], seed_genres: List[str], 
                            target_features: Dict, limit: int = 20) -> List[Dict]:
        """Search for similar tracks using search API (fallback from recommendations)"""
        try:
            tracks = []
            
            # Method 1: Search by artist names
            for artist_name in seed_artists[:3]:
                try:
                    # Search for tracks by this artist
                    search_query = f"artist:{artist_name}"
                    results = self._make_request_with_retry(
                        self.sp.search,
                        q=search_query,
                        type='track',
                        limit=10
                    )
                    
                    if results and results['tracks']['items']:
                        tracks.extend(results['tracks']['items'])
                        
                except Exception as e:
                    logger.warning(f"Could not search for artist {artist_name}: {e}")
                    continue
            
            # Method 2: Search by genre keywords (simplified)
            for genre in seed_genres[:2]:
                try:
                    # Use genre as search term
                    genre_clean = genre.replace(" ", "+")
                    search_query = f"genre:{genre_clean}"
                    
                    results = self._make_request_with_retry(
                        self.sp.search,
                        q=search_query,
                        type='track',
                        limit=8
                    )
                    
                    if results and results['tracks']['items']:
                        tracks.extend(results['tracks']['items'])
                        
                except Exception as e:
                    logger.warning(f"Could not search for genre {genre}: {e}")
                    continue
            
            # Method 3: Fallback - search for popular tracks in main genre
            if not tracks and seed_genres:
                try:
                    main_genre = seed_genres[0].split()[0]  # Get first word of genre
                    search_query = f"{main_genre} popular"
                    
                    results = self._make_request_with_retry(
                        self.sp.search,
                        q=search_query,
                        type='track',
                        limit=15
                    )
                    
                    if results and results['tracks']['items']:
                        tracks.extend(results['tracks']['items'])
                        
                except Exception as e:
                    logger.warning(f"Fallback search failed: {e}")
            
            # Remove duplicates and limit results
            seen_ids = set()
            unique_tracks = []
            
            for track in tracks:
                if track['id'] not in seen_ids and track['id']:
                    seen_ids.add(track['id'])
                    unique_tracks.append(track)
                    
                    if len(unique_tracks) >= limit:
                        break
            
            logger.info(f"Found {len(unique_tracks)} tracks using search method")
            return unique_tracks
            
        except Exception as e:
            logger.error(f"Error in search_similar_tracks: {e}")
            return []

class MusicProfiler:
    """Analyze music and build user profiles"""
    
    def __init__(self, spotify_client: SimpleSpotifyClient):
        self.spotify = spotify_client
    
    def extract_track_data(self, tracks: List[Dict]) -> List[TrackData]:
        """Extract data from Spotify tracks using only available data"""
        if not tracks:
            return []
        
        logger.info(f"Extracting data from {len(tracks)} tracks")
        
        track_data = []
        processed = 0
        
        for track in tracks:
            if not track.get('id'):
                continue
            
            processed += 1
            if processed % 20 == 0:
                logger.info(f"Processed {processed}/{len(tracks)} tracks")
            
            try:
                # Get artist info (with caching)
                artist_id = track['artists'][0]['id']
                artist_info = self.spotify.get_artist_info(artist_id)
                
                # Extract release year
                release_date = track.get('album', {}).get('release_date', '2000')
                try:
                    release_year = int(release_date[:4])
                except:
                    release_year = 2000
                
                data = TrackData(
                    track_id=track['id'],
                    name=track['name'],
                    artist=track['artists'][0]['name'],
                    album=track.get('album', {}).get('name', ''),
                    popularity=track.get('popularity', 50),
                    duration_ms=track.get('duration_ms', 200000),
                    release_year=release_year,
                    genres=artist_info.get('genres', []) if artist_info else [],
                    artist_popularity=artist_info.get('popularity', 50) if artist_info else 50,
                    explicit=track.get('explicit', False)
                )
                
                track_data.append(data)
                
            except Exception as e:
                logger.warning(f"Error processing track {track.get('name', 'Unknown')}: {e}")
                continue
        
        logger.info(f"Successfully extracted data for {len(track_data)} tracks")
        return track_data
    
    def build_user_profile(self, track_data: List[TrackData]) -> UserTaste:
        """Build user taste profile using available data only"""
        if not track_data:
            return UserTaste({}, {}, {}, 0.0)
        
        # Analyze genres
        genre_counter = Counter()
        for track in track_data:
            for genre in track.genres:
                genre_counter[genre] += 1
        
        total_genre_counts = sum(genre_counter.values())
        favorite_genres = {
            genre: count / total_genre_counts
            for genre, count in genre_counter.most_common(15)
        } if total_genre_counts > 0 else {}
        
        # Analyze artists
        artist_counter = Counter(track.artist for track in track_data)
        total_tracks = len(track_data)
        favorite_artists = {
            artist: count / total_tracks
            for artist, count in artist_counter.most_common(20)
        }
        
        # Analyze available preferences (no audio features)
        preferences = {
            'popularity': statistics.mean([t.popularity for t in track_data]),
            'duration_minutes': statistics.mean([t.duration_ms / 60000 for t in track_data]),
            'release_year': statistics.mean([t.release_year for t in track_data]),
            'artist_popularity': statistics.mean([t.artist_popularity for t in track_data]),
            'explicit_ratio': sum(1 for t in track_data if t.explicit) / total_tracks
        }
        
        # Calculate diversity
        unique_artists = len(set(t.artist for t in track_data))
        diversity_score = unique_artists / total_tracks
        
        return UserTaste(
            favorite_genres=favorite_genres,
            favorite_artists=favorite_artists,
            preferences=preferences,
            diversity_score=diversity_score
        )

class SimpleRecommendationEngine:
    """Simple but effective recommendation engine"""
    
    def __init__(self, spotify_client: SimpleSpotifyClient):
        self.spotify = spotify_client
        self.profiler = MusicProfiler(spotify_client)
    
    def recommend_songs(self, playlist_id: str, num_recommendations: int = 10) -> List[Dict]:
        """Generate song recommendations"""
        logger.info(f"Generating recommendations for playlist: {playlist_id}")
        
        # Get playlist data
        tracks = self.spotify.get_playlist_tracks(playlist_id)
        if len(tracks) < 3:
            logger.warning("Playlist too small")
            return []
        
        track_data = self.profiler.extract_track_data(tracks)
        user_taste = self.profiler.build_user_profile(track_data)
        
        # Generate recommendations using multiple strategies
        recommendations = []
        
        # Strategy 1: Artist-based recommendations (40%)
        artist_recs = self._get_artist_recommendations(user_taste, track_data)
        recommendations.extend(self._score_recommendations(artist_recs, 0.4, "similar_artist"))
        
        # Strategy 2: Genre-based recommendations (35%)
        genre_recs = self._get_genre_recommendations(user_taste)
        recommendations.extend(self._score_recommendations(genre_recs, 0.35, "genre_match"))
        
        # Strategy 3: Feature-based recommendations (25%)
        feature_recs = self._get_feature_recommendations(user_taste)
        recommendations.extend(self._score_recommendations(feature_recs, 0.25, "preference_match"))
        
        # Remove duplicates and existing tracks
        existing_ids = {t.track_id for t in track_data}
        unique_recs = []
        seen_ids = set()
        
        for rec in sorted(recommendations, key=lambda x: x['score'], reverse=True):
            track_id = rec['track']['id']
            if track_id not in seen_ids and track_id not in existing_ids:
                unique_recs.append(rec)
                seen_ids.add(track_id)
                
                if len(unique_recs) >= num_recommendations:
                    break
        
        return unique_recs[:num_recommendations]
    
    def _get_artist_recommendations(self, user_taste: UserTaste, track_data: List[TrackData]) -> List[Dict]:
        """Get recommendations based on favorite artists using multiple approaches"""
        recommendations = []
        top_artists = list(user_taste.favorite_artists.keys())[:5]
        
        # Try search-based approach
        search_tracks = self.spotify.search_similar_tracks(
            seed_artists=top_artists,
            seed_genres=[],
            target_features={'popularity': user_taste.preferences['popularity']},
            limit=15
        )
        recommendations.extend(search_tracks)
        
        # Fallback: Search for "similar to" queries
        if len(recommendations) < 5:
            for artist in top_artists[:2]:
                try:
                    search_query = f"{artist} similar artist"
                    results = self.spotify.sp.search(search_query, type='track', limit=8)
                    if results and results['tracks']['items']:
                        recommendations.extend(results['tracks']['items'])
                except Exception as e:
                    logger.warning(f"Fallback search for {artist} failed: {e}")
                    continue
        
        return recommendations[:15]  # Limit to 15 tracks
    
    def _get_genre_recommendations(self, user_taste: UserTaste) -> List[Dict]:
        """Get recommendations based on favorite genres using search"""
        recommendations = []
        top_genres = list(user_taste.favorite_genres.keys())[:3]
        
        # Try genre-based search
        search_tracks = self.spotify.search_similar_tracks(
            seed_artists=[],
            seed_genres=top_genres,
            target_features={'popularity': user_taste.preferences['popularity']},
            limit=15
        )
        recommendations.extend(search_tracks)
        
        # Fallback: Search for genre keywords directly
        if len(recommendations) < 5:
            for genre in top_genres[:2]:
                try:
                    # Clean up genre name for search
                    genre_search = genre.replace("outlaw ", "").replace("classic ", "")
                    search_query = f"{genre_search} music"
                    
                    results = self.spotify.sp.search(search_query, type='track', limit=8)
                    if results and results['tracks']['items']:
                        recommendations.extend(results['tracks']['items'])
                except Exception as e:
                    logger.warning(f"Fallback genre search for {genre} failed: {e}")
                    continue
        
        return recommendations[:15]
    
    def _get_feature_recommendations(self, user_taste: UserTaste) -> List[Dict]:
        """Get recommendations combining artists and genres"""
        recommendations = []
        
        # Use top artists and genres together
        top_artists = list(user_taste.favorite_artists.keys())[:2]
        top_genres = list(user_taste.favorite_genres.keys())[:2]
        
        search_tracks = self.spotify.search_similar_tracks(
            seed_artists=top_artists,
            seed_genres=top_genres,
            target_features={'popularity': user_taste.preferences['popularity']},
            limit=10
        )
        recommendations.extend(search_tracks)
        
        # Additional fallback: Search for tracks from the same era
        if len(recommendations) < 5:
            try:
                avg_year = int(user_taste.preferences['release_year'])
                decade = (avg_year // 10) * 10
                
                if top_genres:
                    main_genre = top_genres[0].split()[0]  # Get main genre word
                    search_query = f"{main_genre} {decade}s"
                    
                    results = self.spotify.sp.search(search_query, type='track', limit=10)
                    if results and results['tracks']['items']:
                        recommendations.extend(results['tracks']['items'])
                        
            except Exception as e:
                logger.warning(f"Era-based search failed: {e}")
        
        return recommendations[:10]
    
    def _score_recommendations(self, tracks: List[Dict], weight: float, method: str) -> List[Dict]:
        """Score recommendations"""
        scored = []
        for track in tracks:
            score = weight * (0.7 + random.uniform(0, 0.3))  # Add some randomness
            scored.append({
                'track': track,
                'score': score,
                'method': method,
                'reasoning': self._generate_reason(track, method)
            })
        return scored
    
    def _generate_reason(self, track: Dict, method: str) -> str:
        """Generate explanation for recommendation"""
        artist = track['artists'][0]['name']
        
        if method == "similar_artist":
            return f"You listen to {artist} or similar artists"
        elif method == "genre_match":
            return f"Matches your preferred music genres"
        elif method == "preference_match":
            return f"Based on your music taste preferences"
        return "Recommended for you"

class PlaylistManager:
    """Manage saved playlists"""
    
    def __init__(self, filename: str = "saved_playlists.json"):
        self.filename = Path(filename)
        self.playlists = self._load_playlists()
    
    def _load_playlists(self) -> Dict:
        """Load saved playlists"""
        if self.filename.exists():
            try:
                with open(self.filename, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_playlist(self, playlist_id: str, name: str, track_count: int = 0):
        """Save a playlist"""
        self.playlists[name] = {
            'id': playlist_id,
            'saved_at': datetime.now().isoformat(),
            'track_count': track_count
        }
        self._save_playlists()
    
    def _save_playlists(self):
        """Save playlists to file"""
        with open(self.filename, 'w') as f:
            json.dump(self.playlists, f, indent=2)
    
    def get_playlists(self) -> Dict:
        """Get all saved playlists"""
        return self.playlists
    
    def clear_all(self):
        """Clear all saved playlists"""
        self.playlists.clear()
        self._save_playlists()

class SpotifyRecommenderApp:
    """Main application"""
    
    def __init__(self):
        self.spotify = SimpleSpotifyClient()
        self.engine = SimpleRecommendationEngine(self.spotify)
        self.playlist_manager = PlaylistManager()
    
    def run(self):
        """Main app loop"""
        print("\n" + "🎵" * 50)
        print("🎧 SPOTIFY SONG RECOMMENDER 🎧")
        print("🎵" * 50)
        print("✨ Get personalized music recommendations! ✨\n")
        
        while True:
            self._show_menu()
            choice = input("\n🎯 Choose option: ").strip()
            
            try:
                choice_num = int(choice)
                if choice_num == 1:
                    self._recommend_from_url()
                elif choice_num == 2:
                    self._recommend_from_saved()
                elif choice_num == 3:
                    self._save_playlist()
                elif choice_num == 4:
                    self._view_saved_playlists()
                elif choice_num == 5:
                    self._clear_saved_playlists()
                elif choice_num == 6:
                    self._test_connection()
                elif choice_num == 7:
                    print("\n🎵 Thanks for using Spotify Recommender! 🎵")
                    break
                else:
                    print("❌ Invalid option")
            except ValueError:
                print("❌ Please enter a number")
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _show_menu(self):
        """Show main menu"""
        print("\n" + "─" * 40)
        print("📋 MENU")
        print("─" * 40)
        print("1. 🎧 Get recommendations from playlist URL")
        print("2. 💾 Get recommendations from saved playlist")
        print("3. 📁 Save playlist for later")
        print("4. 📋 View saved playlists")
        print("5. 🗑️  Clear saved playlists")
        print("6. 🔧 Test Spotify connection")
        print("7. 🚪 Exit")
        print("─" * 40)
    
    def _recommend_from_url(self):
        """Get recommendations from URL"""
        print("\n🔗 PLAYLIST RECOMMENDATIONS")
        print("─" * 30)
        
        url = input("Enter Spotify playlist URL: ").strip()
        if not url:
            print("❌ No URL provided")
            return
        
        playlist_id = self._extract_playlist_id(url)
        if not playlist_id:
            print("❌ Invalid playlist URL")
            return
        
        self._generate_and_show_recommendations(playlist_id)
    
    def _recommend_from_saved(self):
        """Get recommendations from saved playlist"""
        playlists = self.playlist_manager.get_playlists()
        if not playlists:
            print("\n❌ No saved playlists")
            return
        
        print("\n💾 SAVED PLAYLISTS")
        print("─" * 25)
        
        playlist_list = list(playlists.items())
        for i, (name, data) in enumerate(playlist_list, 1):
            track_count = data.get('track_count', 'Unknown')
            print(f"{i}. {name} ({track_count} tracks)")
        
        try:
            choice = int(input(f"\nSelect playlist (1-{len(playlist_list)}): ")) - 1
            if 0 <= choice < len(playlist_list):
                name, data = playlist_list[choice]
                self._generate_and_show_recommendations(data['id'], name)
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Please enter a number")
    
    def _generate_and_show_recommendations(self, playlist_id: str, playlist_name: str = None):
        """Generate and display recommendations"""
        try:
            print(f"\n🔄 Analyzing playlist...")
            
            if not playlist_name:
                try:
                    playlist_info = self.spotify.sp.playlist(playlist_id)
                    playlist_name = playlist_info['name']
                except:
                    playlist_name = "Unknown Playlist"
            
            recommendations = self.engine.recommend_songs(playlist_id, 8)
            
            if not recommendations:
                print("❌ Could not generate recommendations")
                return
            
            self._display_recommendations(recommendations, playlist_name)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            print(f"❌ Error: {e}")
    
    def _display_recommendations(self, recommendations: List[Dict], playlist_name: str):
        """Display recommendations beautifully"""
        print("\n" + "🎵" * 50)
        print(f"✨ RECOMMENDATIONS FOR: {playlist_name.upper()}")
        print("🎵" * 50)
        
        for i, rec in enumerate(recommendations, 1):
            track = rec['track']
            score = rec['score']
            reason = rec['reasoning']
            
            print(f"\n🎧 {i}. {track['name']}")
            print(f"   👤 {track['artists'][0]['name']}")
            print(f"   💿 {track.get('album', {}).get('name', 'Unknown Album')}")
            print(f"   📊 Score: {score:.2f}")
            print(f"   💭 {reason}")
            
            # Spotify link
            spotify_url = track.get('external_urls', {}).get('spotify', '')
            if spotify_url:
                print(f"   🔗 {spotify_url}")
            
            print("   " + "─" * 40)
        
        print(f"\n🎯 Generated {len(recommendations)} recommendations!")
        
        # Ask to export
        export = input("\n💾 Export to file? (y/n): ").lower().strip()
        if export == 'y':
            self._export_recommendations(recommendations, playlist_name)
    
    def _export_recommendations(self, recommendations: List[Dict], playlist_name: str):
        """Export recommendations to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recommendations_{timestamp}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"🎵 SPOTIFY RECOMMENDATIONS FOR: {playlist_name}\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                
                for i, rec in enumerate(recommendations, 1):
                    track = rec['track']
                    f.write(f"{i}. {track['name']}\n")
                    f.write(f"   Artist: {track['artists'][0]['name']}\n")
                    f.write(f"   Album: {track.get('album', {}).get('name', 'Unknown')}\n")
                    f.write(f"   Reason: {rec['reasoning']}\n")
                    
                    spotify_url = track.get('external_urls', {}).get('spotify', '')
                    if spotify_url:
                        f.write(f"   Spotify: {spotify_url}\n")
                    f.write("\n")
            
            print(f"✅ Exported to {filename}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def _save_playlist(self):
        """Save a playlist"""
        print("\n📁 SAVE PLAYLIST")
        print("─" * 20)
        
        url = input("Enter Spotify playlist URL: ").strip()
        if not url:
            print("❌ No URL provided")
            return
        
        playlist_id = self._extract_playlist_id(url)
        if not playlist_id:
            print("❌ Invalid URL")
            return
        
        try:
            playlist_info = self.spotify.sp.playlist(playlist_id)
            name = playlist_info['name']
            track_count = playlist_info['tracks']['total']
            
            print(f"📊 Found: {name} ({track_count} tracks)")
            
            custom_name = input(f"Save as (Enter for '{name}'): ").strip()
            if custom_name:
                name = custom_name
            
            self.playlist_manager.save_playlist(playlist_id, name, track_count)
            print(f"✅ Saved as '{name}'!")
            
        except Exception as e:
            print(f"❌ Could not save: {e}")
    
    def _view_saved_playlists(self):
        """View all saved playlists"""
        playlists = self.playlist_manager.get_playlists()
        if not playlists:
            print("\n❌ No saved playlists")
            return
        
        print("\n📋 SAVED PLAYLISTS")
        print("─" * 30)
        
        for name, data in playlists.items():
            saved_date = datetime.fromisoformat(data['saved_at']).strftime("%Y-%m-%d")
            track_count = data.get('track_count', 'Unknown')
            print(f"📁 {name}")
            print(f"   🎵 {track_count} tracks")
            print(f"   📅 Saved: {saved_date}")
            print()
    
    def _clear_saved_playlists(self):
        """Clear all saved playlists"""
        playlists = self.playlist_manager.get_playlists()
        if not playlists:
            print("\n❌ No playlists to clear")
            return
        
        print(f"\n🗑️ You have {len(playlists)} saved playlists")
        confirm = input("Clear all? (y/n): ").lower().strip()
        
        if confirm == 'y':
            self.playlist_manager.clear_all()
            print("✅ All playlists cleared!")
        else:
            print("❌ Cancelled")
    
    def _test_connection(self):
        """Test Spotify connection"""
        print("\n🔧 Testing Spotify connection...")
        
        try:
            # Test with a simple search
            results = self.spotify.sp.search("test", type='track', limit=1)
            if results and results['tracks']['items']:
                print("✅ Spotify connection successful!")
            else:
                print("⚠️ Connection works but no results returned")
                
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("💡 Check your .env file with Spotify credentials")
    
    def _extract_playlist_id(self, url: str) -> Optional[str]:
        """Extract playlist ID from URL"""
        # Handle different URL formats
        if 'open.spotify.com/playlist/' in url:
            return url.split('playlist/')[-1].split('?')[0]
        elif 'spotify:playlist:' in url:
            return url.split(':')[-1]
        else:
            # Maybe it's just the ID
            if len(url) == 22 and url.replace('-', '').replace('_', '').isalnum():
                return url
        return None

def create_env_file():
    """Help user create .env file"""
    env_file = Path(".env")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    print("\n🔑 SPOTIFY CREDENTIALS SETUP")
    print("─" * 35)
    print("To use this app, you need Spotify API credentials:")
    print("1. Go to https://developer.spotify.com/dashboard")
    print("2. Create a new app")
    print("3. Copy your Client ID and Client Secret")
    print()
    
    client_id = input("Enter your Spotify Client ID: ").strip()
    client_secret = input("Enter your Spotify Client Secret: ").strip()
    
    if client_id and client_secret:
        with open(env_file, 'w') as f:
            f.write(f"SPOTIFY_CLIENT_ID={client_id}\n")
            f.write(f"SPOTIFY_CLIENT_SECRET={client_secret}\n")
        
        print("✅ .env file created successfully!")
        return True
    else:
        print("❌ Invalid credentials provided")
        return False

def main():
    """Main entry point"""
    print("🚀 Starting Spotify Recommender...")
    
def main():
    """Main entry point"""
    print("🚀 Starting Spotify Recommender...")
    
    # Check for .env file
    if not Path(".env").exists():
        print("\n⚠️ No .env file found")
        if not create_env_file():
            print("❌ Setup failed. Exiting.")
            return
    
    try:
        app = SpotifyRecommenderApp()
        app.run()
        
    except ValueError as e:
        if "credentials" in str(e).lower():
            print(f"\n❌ {e}")
            print("💡 Run the app again to set up credentials")
        else:
            print(f"❌ Error: {e}")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()