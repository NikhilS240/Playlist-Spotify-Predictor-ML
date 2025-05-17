import os
from dotenv import load_dotenv
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
from datetime import datetime
import pandas as pd
import torch
from torch.utils.data import Dataset


print("loading....")


#the first list is only taking the first element
#the 2nd list needs to become a set and also taske multiple elements

load_dotenv()

client_id = os.getenv("client_id")
client_secret = os.getenv("client_secret")


# Setup Spotify API client
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret
))

inputUser = 0


artist_id = "1RyvyyTE3xzB2ZywiAwp0i"

current_year = datetime.now().year


# inputUser = input("Please enter the id: ")
# inputUser = "8uNY3LXNJ2aZUn2sCIlQ6"

playlist_id = "68uNY3LXNJ2aZUn2sCIlQ6"



offset = 0
limit = 100
count = 0
i = 0

list_of_genres = []
list_of_artists = []



while True:

  

    playlist_data = sp.playlist_items(playlist_id, offset=offset, limit=limit)
    items = playlist_data['items']

    if not items:
        break
    
    

    for item in items:
        index_of_genres = []
        artist_genres = []

        index_of_artist = []
        artist_name = []
        

        track = item['track']
        nameTrack = track['name']
        album = track['album']
        length = track['duration_ms']
        release = album['release_date']
        album_type = album['album_type']
        popularity = track['popularity']

        date_list = []
        
        #finds the release of the album
        for i in range(len(release)):
            if release[i] == '-':
                break

            else:
                date_list.append(release[i])

        final_string = "".join(date_list)

        final_int = int(final_string)

        popularity = popularity / 100

        length = round((length /600000), 3)


        if (album_type == 'album'):
            album_type = 1
        

        else:
            album_type = 0

        display_names = []
      
        

        
        for artist in album['artists']:
            
            
            display_names.append(artist['name'])
            
            artist_id = (artist['id'])
            spec_artist_id = sp.artist(artist_id)

            spec_artist_id_name = spec_artist_id['name']




            if spec_artist_id_name not in list_of_artists:
                list_of_artists.append(spec_artist_id_name)

                

            spec_artist_id_genre = spec_artist_id['genres']

            # combined_string = ", ".join(spec_artist_id_genre)

            # split_genres = combined_string.split(", ")

            for genre in spec_artist_id_genre:
                if genre not in list_of_genres:
                    list_of_genres.append(genre)  


            artist_genres.extend(spec_artist_id_genre)


            artist_name.append(spec_artist_id_name)
    

        if artist_genres:
            for genre in artist_genres:
                
                if genre in list_of_genres:
                    
                    index = list_of_genres.index(genre)  # Find the index of each genre in list_of_genres
                    index_of_genres.append(index)

        else:
            index_of_genres.append(-1)


        #artist_name stores the current local small list i think
        for artist in artist_name:
            
            if artist in list_of_artists:
                
                index = list_of_artists.index(artist)  # Find the index of each genre in list_of_genres
                index_of_artist.append(index)

  

        final_int = current_year - final_int
        final_int = final_int / 10


        #remove the name 
        combined = [

        index_of_genres,            # already a list
        index_of_artist,            # already a list
        [album_type],               # scalar wrapped in list
        [popularity],               # scalar wrapped in list
        [length],                   # scalar wrapped in list
        [final_int]                 # scalar wrapped in list
    ]
        combined_df = pd.DataFrame(combined)
        # print(combined_df)
        print(combined)


        # print(f"{name} {display_names} {album_type} {popularity} {length} {release} ")
        count = count + 1

        
#for non-numerical values i could either use frequency encoidig 
#one-hot encoding 
#embedding layer

#give each thing a number asnd increment

    offset = offset + limit


   
    ##this needs to be in the for loop of the other stuff //rn it just takes the last vaue
    ##fix the last element prbolem  
    class MyDataset(Dataset):
        def __init__(self, index_of_genres, index_of_artist, album_type):
            # data and labels should be tensors or lists of equal length
            self.index_of_genres = torch.tensor(index_of_genres, dtype=torch.float32)
            self.index_of_artist = torch.tensor(index_of_artist, dtype=torch.long)
            self.album_type = torch.tensor(album_type, dtype=torch.long)

        # def __len__(self):
        #     # Return the total number of samples
        #     return len(self.data)

        def __getitem__(self, idx):
            return {
            'genre': self.index_of_genres,
            'artist': self.index_of_artist,
            'album': self.album_type,
        }

    #try to get the tensor working first

    ds = MyDataset(index_of_genres, index_of_artist, album_type)
    print(ds[0])



print(count)

#THIS PRINTS ALL THE JUNK
# print(json.dumps(dog, indent=7))

