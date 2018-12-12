---
title: Create Spotify Song Pool
notebook: create_song_pool.ipynb
nav_include: 3
---

This Jupyter notebook creates a pool containing unique songs obtained through Spotify's API. It requests data for all unique songs found in a subset of Spotify's 1 Million Playlist Set. The subset consists of 10,000 playlists. The number of unique songs found in this subset and pushed into a pool csv file is approximaetly 170,000. This, we assume, is a sufficiently large enough pool for recommending songs to a playlist. Song features are added to the pool in order to be used as a source of information.

<hr style="height:2pt">


We import a set of functions we created in order to make notebook codes easier to read. These functions, stored in a .py file called "spotify_api_fuction_set", are used for handling a Library that communicates with the Spotify API called Spotipy. The Spotipy library can be found here (https://spotipy.readthedocs.io/en/latest/). Note that the functions created are specific to this project (See EDA section for list of functions inside this .py file).



```python
import spotify_api_function_set as sps #imports set of functions created to use spotify API
```


We load a subset of 10,000 playlists from the 1 Million Playlist Dataset from Spotify using the json library.



```python
path = 'data'
file_names = ["mpd.slice.0-999", "mpd.slice.1000-1999", "mpd.slice.2000-2999",
              "mpd.slice.3000-3999", "mpd.slice.4000-4999", "mpd.slice.5000-5999",
              "mpd.slice.6000-6999", "mpd.slice.7000-7999", "mpd.slice.8000-8999", "mpd.slice.9000-9999"]

spotify_playlist = []
for file in file_names:
    with open(path+"/"+file+".json", "r") as fd:
        plylist_temp = json.load(fd)
        plylist_temp = plylist_temp.get('playlists')
        spotify_playlist = spotify_playlist + plylist_temp
```


We define the number of playlists we wish to use as a source for song pool generation. In this case we will use all 10,000 playlists. From here, for each playlist, we extract each song's Uniform Resource Identifier (URI) and each song's artist URI so we can use it later with Spotify's API.



```python
N = 10000 #Number of playlists to request

track_uri = []
artist_uri = []

for i in range(N):
    track_id = sps.get_playlist_n(spotify_playlist[i], feature = 'track_uri', n_playlist = i)
    artist_id = sps.get_playlist_n(spotify_playlist[i], feature = 'artist_uri', n_playlist = i)  

    track_uri.extend(track_id)
    artist_uri.extend(artist_id)
```


Since we expect many songs to be repeated from playlist to playlist, we store the track and artist URIs in a pandas dataframe in order to drop any duplicates based on track URIs.



```python
data = [np.array(track_uri).T, np.array(artist_uri).T]
data = np.transpose(data)
temp_df = pd.DataFrame(data)
temp_df.columns = ['track_uri', 'artist_uri']
```


We check the length of the dataframe containing all songs extracted from the 10,000 playlists. We see that there are currently 664,712 songs in the dataframe.



```python
len(temp_df)
```





    664712



Dropping duplicated songs, we reduce the playlist to 170,089 unique songs. We do this before requesting API information in order to prevent unnecessary requests.



```python
temp_df = temp_df.drop_duplicates(subset='track_uri') #Remove duplicates
len(temp_df)
```





    170089





```python
track_uri = list(temp_df.track_uri)
artist_uri = list(temp_df.artist_uri)
sp = sps.create_spotipy_obj() #create spotify object to use to request songs
```


We request song and artist features provided by spotify's API for all unique songs found in the 10,000 playlist subset. We time it to get a sense of speed. Note, this code took us approximately 22 minutes to run. Feel free to use a smaller Playlist subset (N above) to test the code first.



```python
start_time = time.time()
t_features, a_features = sps.get_all_features(track_uri, artist_uri, sp)
print("--- %s seconds ---" % (time.time() - start_time))
```


    --- 1293.6080300807953 seconds ---




```python
data = [np.array(t_features).T, np.array(a_features).T]
data = np.transpose(data)
feature_pd = pd.DataFrame(data)
feature_pd.columns = ['t_features', 'a_features']
```


Before proceeding any further, we check to see if Spotify returned any NonType objects and drop them. When we ran the code, we got only one NonType object for a song, hence our pool was reduced by one song.



```python
feature_pd = feature_pd.dropna()
t_features = list(feature_pd.t_features)
a_features = list(feature_pd.a_features)
```


We create a pandas dataframe containing unique songs with its features and categorize each song into a genre just like we did when doing data exploration and preparation. We also timed this step, fortunately for the 10,000 playlists, this took about 5 minutes.



```python
songs_df = sps.create_song_df(t_features, a_features, list(range(len(t_features))))
```




```python
start_time = time.time()
songs_df_unique = sps.genre_generator(songs_df)
print("--- %s seconds ---" % (time.time() - start_time))
```

    --- 265.5861220359802 seconds ---


We clean the data a bit further.



```python
cols = ['song_uri', 'duration_ms', 'time_signature', 'key', 'tempo',
       'energy', 'mode', 'loudness', 'speechiness', 'danceability',
       'acousticness', 'instrumentalness', 'valence', 'liveness',
       'artist_followers', 'artist_name', 'artist_popularity', 'artist_uri','genre']
drop = set(cols)^set(songs_df_unique.columns)
```




```python
pool_df = songs_df_unique.drop(drop, axis=1)
```


Check to see if things look ok.



```python
pool_df.head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>song_uri</th>
      <th>duration_ms</th>
      <th>time_signature</th>
      <th>key</th>
      <th>tempo</th>
      <th>energy</th>
      <th>mode</th>
      <th>loudness</th>
      <th>speechiness</th>
      <th>danceability</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>valence</th>
      <th>liveness</th>
      <th>artist_followers</th>
      <th>artist_uri</th>
      <th>artist_name</th>
      <th>artist_popularity</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>spotify:track:0UaMYEvWZi0ZqiDOoHU3YI</td>
      <td>226864</td>
      <td>4</td>
      <td>4</td>
      <td>125.461</td>
      <td>0.813</td>
      <td>0</td>
      <td>-7.105</td>
      <td>0.1210</td>
      <td>0.904</td>
      <td>0.03110</td>
      <td>0.006970</td>
      <td>0.810</td>
      <td>0.0471</td>
      <td>909185</td>
      <td>spotify:artist:2wIVse2owClT7go1WT98tk</td>
      <td>Missy Elliott</td>
      <td>76</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>spotify:track:6I9VzXrHxO9rA9A5euc8Ak</td>
      <td>198800</td>
      <td>4</td>
      <td>5</td>
      <td>143.040</td>
      <td>0.838</td>
      <td>0</td>
      <td>-3.914</td>
      <td>0.1140</td>
      <td>0.774</td>
      <td>0.02490</td>
      <td>0.025000</td>
      <td>0.924</td>
      <td>0.2420</td>
      <td>5455441</td>
      <td>spotify:artist:26dSoYclwsYLMAKD3tpOr4</td>
      <td>Britney Spears</td>
      <td>82</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spotify:track:0WqIKmW4BTrj3eJFmnCKMv</td>
      <td>235933</td>
      <td>4</td>
      <td>2</td>
      <td>99.259</td>
      <td>0.758</td>
      <td>0</td>
      <td>-6.583</td>
      <td>0.2100</td>
      <td>0.664</td>
      <td>0.00238</td>
      <td>0.000000</td>
      <td>0.701</td>
      <td>0.0598</td>
      <td>16678709</td>
      <td>spotify:artist:6vWDO969PvNqNYHIOW5v0m</td>
      <td>Beyoncé</td>
      <td>87</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>spotify:track:1AWQoqb9bSvzTjaLralEkT</td>
      <td>267267</td>
      <td>4</td>
      <td>4</td>
      <td>100.972</td>
      <td>0.714</td>
      <td>0</td>
      <td>-6.055</td>
      <td>0.1400</td>
      <td>0.891</td>
      <td>0.20200</td>
      <td>0.000234</td>
      <td>0.818</td>
      <td>0.0521</td>
      <td>7341126</td>
      <td>spotify:artist:31TPClRtHm23RisEBtV3X7</td>
      <td>Justin Timberlake</td>
      <td>83</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>4</th>
      <td>spotify:track:1lzr43nnXAijIGYnCT8M8H</td>
      <td>227600</td>
      <td>4</td>
      <td>0</td>
      <td>94.759</td>
      <td>0.606</td>
      <td>1</td>
      <td>-4.596</td>
      <td>0.0713</td>
      <td>0.853</td>
      <td>0.05610</td>
      <td>0.000000</td>
      <td>0.654</td>
      <td>0.3130</td>
      <td>1044532</td>
      <td>spotify:artist:5EvFsr3kj42KNv97ZEnqij</td>
      <td>Shaggy</td>
      <td>74</td>
      <td>rap</td>
    </tr>
  </tbody>
</table>
</div>



Finally we store the pool into the specified path, we drop the index as it isn't necesarry.



```python
pool_df.to_csv(path+'/'+'big_song_pool.csv', index=False)
```
