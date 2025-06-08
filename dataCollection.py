import os
from dotenv import load_dotenv
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
import torch
from torch.utils.data import Dataset
import torch.nn as nn

import math
import random
import sys
from torch.utils.data import DataLoader
from spotipy.exceptions import SpotifyException


from collections import Counter
artist_frequency = Counter()

Running = True


print("---------------------------------------")
print("Welcome to the Spotify Song recommender")
print("---------------------------------------")


load_dotenv()
client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")

        # Setup Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret
        ))



def my_function(name):
            

            current_year = datetime.now().year
            playlist_id = name

            offset = 0
            limit = 100
            count = 0

            list_of_genres = []
            list_of_artists = []

            all_index_of_genres = []
            all_index_of_artist = []
            all_album_type = []
            all_popularity = []
            all_length = []
            all_final_int = []
          

            while True:
                try:
                    sys.stderr = open(os.devnull, 'w')
                    playlist_data = sp.playlist_items(playlist_id, offset=offset, limit=limit)
                    items = playlist_data['items']
                    # sys.stderr.close()
                    # sys.stderr = sys.__stderr__

                except Exception:
                    sys.stderr = sys.__stderr__
                    return "false"

                finally:
                    sys.stderr.close()
                    sys.stderr = sys.__stderr__  # always restore stderr
         

                if not items:
                    break

                for item in items:
                    index_of_genres = []
                    artist_genres = []
                    index_of_artist = []
                    artist_name = []
                    track_compare = []

                    track = item['track']
                    if 'album' in track:
                        album = track['album']
                    else:
                        print("Warning: 'album' key missing in track:", track)
                        continue  # or handle appropriately
                    length = track['duration_ms']
                    release = album['release_date']
                    album_type = album['album_type']
                    popularity = track['popularity']
                    track_compare = track['name']

                    year_str = release.split("-")[0]
                    final_int = int(year_str)

                    popularity = popularity / 100
                    length = round((length / 600000), 3)
                    album_type = 1 if album_type == 'album' else 0

                    for artist in album['artists']:
                        artist_id = artist['id']
                        spec_artist = sp.artist(artist_id)
                        spec_artist_name = spec_artist['name']

                        if spec_artist_name not in list_of_artists:
                            list_of_artists.append(spec_artist_name)

                        spec_artist_genres = spec_artist['genres']
                        for genre in spec_artist_genres:
                            if genre not in list_of_genres:
                                list_of_genres.append(genre)

                        artist_genres.extend(spec_artist_genres)
                        artist_name.append(spec_artist_name)
                        artist_frequency[spec_artist_name] += 1


                 
                    if artist_genres:
                        for genre in set(artist_genres):
                            if genre in list_of_genres:
                                idx = list_of_genres.index(genre)
                                index_of_genres.append(idx)
                    else:
                        index_of_genres.append(9999)  # fallback unknown genre

                    # Artist indices
                    for artist in artist_name:
                        if artist in list_of_artists:
                            idx = list_of_artists.index(artist)
                            index_of_artist.append(idx)

                    final_int = current_year - final_int
                    final_int = final_int / 10

                    all_index_of_genres.append(index_of_genres)
                    all_index_of_artist.append(index_of_artist)
                    all_album_type.append(album_type)
                    all_popularity.append(popularity)
                    all_length.append(length)
                    all_final_int.append(final_int)

                    count += 1

                offset += limit

            #.#print("Total tracks processed:", count)


            if artist_frequency:
                most_common_artist, countArtist = artist_frequency.most_common(1)[0]
                #.#print("Most common artist:", most_common_artist)


            num_total_genres = len(list_of_genres)
            num_total_artists = len(list_of_artists)

            def indices_to_multi_hot(indices, num_classes):
                multi_hot = [0] * num_classes
                for idx in indices:
                    if 0 <= idx < num_classes:
                        multi_hot[idx] = 1
                return multi_hot

            all_genres_multi_hot = [indices_to_multi_hot(g, num_total_genres) for g in all_index_of_genres]
            all_artists_multi_hot = [indices_to_multi_hot(a, num_total_artists) for a in all_index_of_artist]

            class MyDataset(Dataset):
                def __init__(self, genres_multi_hot, artists_multi_hot, album_type, popularity, length, final_int):
                    self.genres = torch.tensor(genres_multi_hot, dtype=torch.float32)    # multi-hot genres
                    self.artists = torch.tensor(artists_multi_hot, dtype=torch.float32)  # multi-hot artists
                    self.album_type = torch.tensor(album_type, dtype=torch.long)
                    self.popularity = torch.tensor(popularity, dtype=torch.float32)
                    self.length = torch.tensor(length, dtype=torch.float32)
                    self.final_int = torch.tensor(final_int, dtype=torch.float32)

                def __len__(self):
                    return len(self.genres)

                def __getitem__(self, idx):
                    return {
                        'genre': self.genres[idx],       # multi-hot vector
                        'artist': self.artists[idx],     # multi-hot vector
                        'album': self.album_type[idx],
                        'popularity': self.popularity[idx],
                        'length': self.length[idx],
                        'years_old': self.final_int[idx]
                    }

            ds = MyDataset(all_genres_multi_hot, all_artists_multi_hot, all_album_type, all_popularity, all_length, all_final_int)
            #.#print(ds[0])

            #size of the element, num of element


            genre_embedding_dim = 16
            artist_embedding_dim = 16

            genre_emb_layer = nn.Linear(len(list_of_genres), genre_embedding_dim)
            artist_emb_layer = nn.Linear(len(list_of_artists), artist_embedding_dim)



            # Example: get the first batch from your dataset (or create a dummy batch)
            batch_size = 8  # example batch size


            #it should be for i in range of the number of songs
            batch_genres = torch.stack([ds[i]['genre'] for i in range(batch_size)])  # shape: (batch_size, num_genres)
            batch_artists = torch.stack([ds[i]['artist'] for i in range(batch_size)])  # shape: (batch_size, num_artists)

 
            genre_embeddings = genre_emb_layer(batch_genres)  # shape: (batch_size, embedding_dim)
            artist_embeddings = artist_emb_layer(batch_artists)  # shape: (batch_size, embedding_dim)

            #.# print("Genre embeddings shape:", genre_embeddings.shape)
            #.# print("Artist embeddings shape:", artist_embeddings.shape)


            # Example other features batch (grab from dataset similarly)
            batch_album = torch.tensor([ds[i]['album'] for i in range(batch_size)], dtype=torch.float32).unsqueeze(1)
            batch_popularity = torch.tensor([ds[i]['popularity'] for i in range(batch_size)], dtype=torch.float32).unsqueeze(1)
            batch_length = torch.tensor([ds[i]['length'] for i in range(batch_size)], dtype=torch.float32).unsqueeze(1)
            batch_years_old = torch.tensor([ds[i]['years_old'] for i in range(batch_size)], dtype=torch.float32).unsqueeze(1)

 
            combined_features = torch.cat([
                genre_embeddings,
                artist_embeddings,
                batch_album,
                batch_popularity,
                batch_length,
                batch_years_old
            ], dim=1)  # shape: (batch_size, embedding_dim*2 + 4)

            #.#print("Combined input shape:", combined_features.shape)



            ####understand what this does
            input_dim = combined_features.shape[1]

            class SimpleSongModel(nn.Module):
                def __init__(self, input_dim, hidden_dim=64, output_dim=6):  # adjust output_dim as needed
                    super(SimpleSongModel, self).__init__()
                    self.fc1 = nn.Linear(input_dim, hidden_dim)
                    self.relu = nn.ReLU()
                    self.fc2 = nn.Linear(hidden_dim, output_dim)

                def forward(self, x):
                    x = self.fc1(x)
                    x = self.relu(x)
                    x = self.fc2(x)
                    return x

            model = SimpleSongModel(input_dim)
            output = model(combined_features)

            #.#print("Model output shape:", output.shape)
            #.#print("Model output:", output)
            ################


            class_out = output[:, :3]  # First 3 for classification
            reg_out = output[:, 3:]    # Last 3 for regression


            #might need to make this loop for all elements 

       
            lossClass = nn.CrossEntropyLoss()
            targetClass = torch.tensor([ds[i]['album'] for i in range(batch_size)], dtype=torch.long)
            outputClass = lossClass(class_out, targetClass)

           
            lossNum = nn.MSELoss()
            targetNum = torch.tensor([
                [ds[i]['popularity'], ds[i]['length'], ds[i]['years_old']]
                for i in range(batch_size)
            ], dtype=torch.float32)
            outputNum = lossNum(reg_out, targetNum)

            # Final combined loss
            loss = outputClass + outputNum

            #.#print(loss)



  

            train_loader = DataLoader(ds, batch_size=32, shuffle=True)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


            # i need a loop for epoch # and then start training of training of 
            num_epochs = 10

            ##########
            for epoch in range(num_epochs):
                total_loss = 0.0
                for batch in train_loader:
                    # Get batch inputs
                    genre_input = genre_emb_layer(batch['genre'])
                    artist_input = artist_emb_layer(batch['artist'])
                    album_input = batch['album'].unsqueeze(1).float()
                    pop_input = batch['popularity'].unsqueeze(1)
                    len_input = batch['length'].unsqueeze(1)
                    years_input = batch['years_old'].unsqueeze(1)

                    combined = torch.cat([
                        genre_input,
                        artist_input,
                        album_input,
                        pop_input,
                        len_input,
                        years_input
                    ], dim=1)

                    # Forward pass
                    output = model(combined)
                    class_out = output[:, :3]
                    reg_out = output[:, 3:]

               
                    target_class = batch['album']
                    target_reg = torch.stack([
                        batch['popularity'],
                        batch['length'],
                        batch['years_old']
                    ], dim=1)

                
                    loss_class = lossClass(class_out, target_class)
                    loss_reg = lossNum(reg_out, target_reg)
                    loss = loss_class + loss_reg

                    # Backprop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                #.#print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

            #########





            # After training loop ends:

            model.eval()  # set model to eval mode (turn off dropout, etc)

            all_outputs = []

            with torch.no_grad():
                for batch in train_loader:  # or a DataLoader for your full dataset
                    # Prepare inputs (embedding layers + other features)
                    genre_input = genre_emb_layer(batch['genre'])
                    artist_input = artist_emb_layer(batch['artist'])
                    album_input = batch['album'].unsqueeze(1).float()
                    pop_input = batch['popularity'].unsqueeze(1)
                    len_input = batch['length'].unsqueeze(1)
                    years_input = batch['years_old'].unsqueeze(1)

                    combined = torch.cat([
                        genre_input,
                        artist_input,
                        album_input,
                        pop_input,
                        len_input,
                        years_input
                    ], dim=1)

                    output = model(combined)  # (batch_size, 6)
                    all_outputs.append(output)

             
                all_outputs = torch.cat(all_outputs, dim=0)  # (total_dataset_size, 6)

            #.#print("Final output vectors for all dataset items:")
            #.#print(all_outputs)

           
            mean_output = torch.mean(all_outputs, dim=0)
            #.#print("Mean of all output vectors:")
            #.#print(mean_output)







            mean_class = mean_output[:3]    
            mean_reg = mean_output[3:]        
  
            predicted_album_type = torch.argmax(mean_class).item()  



            predicted_popularity = mean_reg[0].item() * 100  
            
            predicted_length = mean_reg[1].item() * 600000  
            predicted_length = abs(predicted_length)
            predicted_years_old = mean_reg[2].item() * 10   
            predicted_release_year = datetime.now().year - int(predicted_years_old)
            predicted_release_year = abs(predicted_release_year)
            predicted_popularity = abs(predicted_popularity)



          
            avg_genre_weights = torch.mean(torch.stack([ds[i]['genre'] for i in range(len(ds))]), dim=0)
            avg_artist_weights = torch.mean(torch.stack([ds[i]['artist'] for i in range(len(ds))]), dim=0)

 
            if avg_genre_weights.numel() >= 3:
                top_genre_indices = torch.topk(avg_genre_weights, 3).indices.tolist()
            else:
   
                top_genre_indices = []

            if avg_artist_weights.numel() >= 3:
                top_artist_indices = torch.topk(avg_artist_weights, 3).indices.tolist()
            else:
           
                top_artist_indices = []


            predicted_genres = [list_of_genres[i] for i in top_genre_indices]
            predicted_artists = [list_of_artists[i] for i in top_artist_indices]



            print("\n--- Predicted Next Song Attributes ---")
            if (predicted_album_type == 1 or predicted_album_type == 0): 
                print("Likely Album Type:", ["single", "album"][predicted_album_type])
            else:
                predicted_album_type == 1
            print("Predicted Popularity:", round(predicted_popularity))
            print("Predicted Length (ms):", int(predicted_length))
            print("Predicted Release Year:", predicted_release_year)
            print("Predicted Genres:", predicted_genres)
            print("Predicted Artists:", predicted_artists)



            match = []
            match_count = 0 

            artist_percentage = (countArtist / count)

            min_bias = 1.01
            max_bias = 2.5

            bias = min_bias + (max_bias - min_bias) * artist_percentage



            for i in range(3):
                match_count = 0

                results = sp.search(q=f'artist:{predicted_artists[i]}', type='artist', limit=1)
                if results['artists']['items']:
                    artist_id = results['artists']['items'][0]['id'] ##id of first artist?
                    artist_genres = results['artists']['items'][0]['genres']  # This is a list

                    # match_count = 0 

                    for predicted in predicted_genres:
                        for actual in artist_genres:
                            if predicted == actual:
                                match_count += 1
                    
                    if predicted_artists[i] == most_common_artist:
                        match_count = match_count * bias

                    
                    match.append(match_count)

                else:
                    # no artist found, append 0 or handle as you want
                    match.append(0)



            totalSum = sum(match)
            randTen = random.random() * totalSum





            #.#print("BIG BREAK")


            cumulative_sum = 0
            artist_name = None

            for i in range(3):
                cumulative_sum += match[i]
                if randTen <= cumulative_sum:
                    results = sp.search(q=f'artist:{predicted_artists[i]}', type='artist', limit=1)
                    if results['artists']['items']:
                        artist_name = results['artists']['items'][0]['name']
                        artist_idx = results['artists']['items'][0]['id']
                    break





            #.#print(artist_name)




            albumChance = 0

            if (predicted_album_type == 0):
                albumChance = 0.4
                #.#print("ddd")


            else:
                albumChance = 0.9


            randAlbum = random.random()
            #.#print(randAlbum)
                            
            albums = []
            offset = 0


            if randAlbum < albumChance:
                album_type = 'album'
            else:
                album_type = 'single'

            while True:
                batch = sp.artist_albums(artist_idx, album_type=album_type, limit=50, offset=offset)
                items = batch['items']
                albums += items 
                if len(items) < 50:
                    break
                offset += 50





            release = album["release_date"]

            album_songs = []

            year_str = release.split("-")[0]
            final_int = int(year_str)


            j = 0





            i = 1
            max_i = 5  # maximum number of expansions
            albums_found = False

            while i <= max_i:
                lower_bound = predicted_release_year - 2 * i
                upper_bound = predicted_release_year + 2 * i

                matched_albums = []
                for album in albums:
                    release = album["release_date"]
                    year_str = release.split("-")[0]
                    final_int = int(year_str)

                    if lower_bound <= final_int <= upper_bound:
                        matched_albums.append(album)

                if matched_albums:
                    for name in matched_albums:


                      

                        #.#print(name)
                        if album_type == 'single':
                            #we needa append only the track
                            album_songs.append(name['name'])
                        # album_songs.append(name)
                    albums_found = True
                    break  # stop expansion once you find matches in this ith range

                i += 1

                

            if not albums_found:
                print("No albums found within extended range.")






 
            if album_type == 'album':
                for album_obj in matched_albums:
                    album_name = album_obj['name']  # extract string name (just added this)
                    album = sp.search(q=f"album:{album_name}", type='album', limit=1)['albums']['items'][0]
                    album_id = album['id']
                    tracks = sp.album_tracks(album_id)['items']
                    
                    for track in tracks:
                        #.#print(track['name'])
                        #we needa append only the track (removed name just now)
                        album_songs.append(track['name'])


            #.#print(album_songs)





        
            predicted_length
            predicted_popularity



       

            d = []

            for track in album_songs:
                song_name = track  # assuming album_songs is a list of song name strings
                try:
                    search_results = sp.search(q=f'track:{song_name}', type='track', limit=1)
                    items = search_results['tracks']['items']
                    if not items:
                        print(f"No results for {song_name}")
                        continue
                    
                    full_track = items[0]
                    song_length = full_track['duration_ms'] / 1000  # convert to seconds
                    song_popularity = full_track['popularity']

                    distance = math.sqrt((predicted_length - song_length) ** 2 +
                                        (predicted_popularity - song_popularity) ** 2)
                    
                    d.append((distance, track))
                    #.#print(f"{song_name}: distance = {distance}")

                except SpotifyException as e:
                    print(f"Spotify API error on {song_name}: {e}")
                    continue


            d.sort(key=lambda x: x[0], reverse=True)


            # print(d)

            num = random.randint(0, 2)
            

            Song = True



            while Song:
                for song in track_compare:
                    if (d[num][1]).strip() == (song).strip():
                        print("The same")
                        num = num + 1
                        break
                else:
                    Song = False



            #some randomization herer is needed!!!!!!!!!!!!!!!!!

            
      
            song_name = d[num][1]

            query = f"artist:{artist_name} track:{song_name}"
            result = sp.search(q=query, type='track', limit=1)
            items = result['tracks']['items']

            if not items:
                print("No matching track found.")
            else:
                track = items[0]
                album_nameGood = track['album']['name']




            print(d[num][1] + " by " + artist_name + " on " + album_nameGood)

            query = f"track:{d[num][1]} artist:{artist_name}"
            results = sp.search(q=query, type='track', limit=1)

    
            if results['tracks']['items']:
                track = results['tracks']['items'][0]
                track_id = track['id']
                url = f"https://open.spotify.com/track/{track_id}"
                print(f"🎧 Listen on Spotify: {url}")
            else:
                print("Spotify Link Not Available.")



while Running: 
    
    n = -1
    print("What do you want to do: \n 1. Play a playlist using a Spotify Link\n 2. Play a playlist after loading it \n 3. Load a song for later playback \n 4. Clear database \n 5. Exit")
    numChoi = (input("What would you like: "))
    if numChoi.isdigit():
        numChoice = int(numChoi)
    else:
        continue

    if (numChoice == 4):
        with open("database.txt", "w") as file:
            pass  # The file is now empty
        print("It has been cleared")

    if (numChoice == 3):
        songLi = input("Please provide a Spotify playlist link (found at top of url): ")

        try:
            sys.stderr = open(os.devnull, 'w')
            playlist = sp.playlist(songLi)
            sys.stderr.close()
            sys.stderr = sys.__stderr__
            n = n + 1

        # Only write to file if the playlist was retrieved successfully
            with open("database.txt", "a") as f:
                f.write(songLi + "\n")

        except Exception:
            sys.stderr = sys.__stderr__
            print("Invalid playlist ID, not saved.")

        


##remove some code
    if (numChoice == 2):

  
        with open("database.txt", "r") as f:
            f.seek(0)
            lines = f.readlines()
        print("\n\nThese are your saved playlists.")
        for line in lines:
            playlist_id = line.strip()


            playlist = sp.playlist(playlist_id)

            count = len(lines) - 1

            print(playlist['name'], count)

        
        user = input("Please provide the number of the stored ID you wish to use: ")
        if user.isdigit():
            i = int(user)

            third_line = lines[i].strip()  

            # print(third_line)
            
            my_function(third_line)
        else:
            continue

        

    if (numChoice == 5):
        
        Running = False

    if (numChoice == 1):
        songLink = input("Please provide a Spotify playlist link (found at top of url): ")
        returE = my_function(songLink)

        if (returE == 'false'):
            my_function(songLink)
        

  
