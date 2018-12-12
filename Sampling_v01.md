---
title: Creating balanced dataset
notebook: Sampling_v01.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}


## Description (summary):

The goal of this section is to create a final dataset of playlists (our sample), with independent variables (tracks and artists features) and the dependent variable (genre of the playlist). Most importantly, we made sure that our sample was equally distributed in each of the classes, since this is important in fitting the models to the training dataset. In order to do so, we had to carry out a number of steps, which included:

    - Requesting playlist IDs, tracks and artist features from Spotify's API using Spotipy Package
    - Setting up a pandas dataframe at the track level
    - Classifying each song to one of 5 genres (rock, pop, poprock, rap, and others)
    - Collapsing the songs to unique playlist IDs, so that for each playlist we would have a vector of average of the features of songs belonging to a playlist, which characterizes each playlist
    - Classifying each playlist to one of 5 genres (rock, pop, poprock, rap, and others), according to the genre most frequent in that given playlist
    - Setting up final sample of playlists of equally distributed in each of the classes (genres)

<hr style="height:2pt">

## Requesting playlist IDs, tracks and artist features from Spotify's API using Spotipy Package

The following function takes a number of playlist and returns the features of the tracks of those selected playlists:



```python
def feature_list_func(plylist_dic, feature, n_playlist, first_pid):
    """"
    This function takes a number of playlist and returns the features of the tracks of those selected playlists.

    input:
        1 - plylist_dic: dictionary of all playlists (dataset in dictionary format: json)
        2 - feature: feature to be selected from each songs in selected playlists
        3 - n_playlists: number of playlists to be selected

    output: list of observations for the feature chosen, for all of the tracks that belong to the selected playlists

    """
    feature_list = []
    pid_list = []
    length_playlist = np.minimum(n_playlist,len(plylist_dic)) # the output will be based on the min of the n_playlist and the actual length of the input playlist
    for i in range(length_playlist):
        playlist = plylist_dic[first_pid + i]
        playlist_pid = playlist.get('pid')
        for j in range(len(playlist.get('tracks'))):
            feature_list.append(playlist.get('tracks')[j].get(feature))
            pid_list.append(playlist_pid)
    return pid_list, feature_list
```


The following code calls the functions above, in order to get the playlist IDs, the track and artist URIs, which will be used later to request the features that will comprise our dataframe.



```python
pid_t, track_uri = feature_list_func(plylist, feature = 'track_uri', n_playlist = 10, first_pid = 0)
pid_a, artist_uri = feature_list_func(plylist, feature = 'artist_uri', n_playlist = 10, first_pid = 0)
```


After getting the URI of the tracks and artists, we requested their features from API Spotify, to create a pandas database at the track level. We used Spotipy API. The Spotify Package can be found at: https://spotipy.readthedocs.io



```python
def create_spotipy_obj():

    """
    Uses dbarjum's client id for DS Project
    """

    SPOTIPY_CLIENT_ID = '54006da9bd7849b7906b944a7fa4e29d'
    SPOTIPY_CLIENT_SECRET = 'f54ae294a30c4a99b2ff330a923cd6e3'
    SPOTIPY_REDIRECT_URI = 'http://localhost/'

    username = 'dbarjum'
    scope = 'user-library-read'

    token = util.prompt_for_user_token(username,scope,client_id=SPOTIPY_CLIENT_ID,
                           client_secret=SPOTIPY_CLIENT_SECRET,
                           redirect_uri=SPOTIPY_REDIRECT_URI)
    client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID,
                                                          client_secret=SPOTIPY_CLIENT_SECRET, proxies=None)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    return sp
```




```python
sp = create_spotipy_obj()
```




```python
def get_all_features(track_list = list, artist_list = list, sp=None):

    """
    This function takes in a list of tracks and a list of artists, along
    with a spotipy object and generates two lists of features from Spotify's API.

    inputs:
        1. track_list: list of all tracks to be included in dataframe
        2. artist_list: list of all artists corresponding to tracks
        3. sp: spotipy object to communicate with Spotify API

    returns:
        1. track_features: list of all features for each track in track_list
        2. artist_features: list of all artist features for each artist in artist_list
    """

    track_features = []
    artist_features = []

    track_iters = int(len(track_list)/50)
    track_remainders = len(track_list)%50

    start = 0
    end = start+50

    for i in range(track_iters):
        track_features.extend(sp.audio_features(track_list[start:end]))
        artist_features.extend(sp.artists(artist_list[start:end]).get('artists'))
        start += 50
        end = start+50


    if track_remainders:
        end = start + track_remainders
        track_features.extend(sp.audio_features(track_list[start:end]))
        artist_features.extend(sp.artists(artist_list[start:end]).get('artists'))

    return track_features, artist_features
```




```python
start_time = time.time()
t_features, a_features = get_all_features(track_uri, artist_uri, sp)
print("--- %s seconds ---" % (time.time() - start_time))
```


    --- 2.8701727390289307 seconds ---


## Setting up a pandas dataframe at the track level

The following function takes in the lists of track and artist features, and generates a dataframe of the features. It also creates columns in the dataframe that represent the genres provided for the artist of each track. These columns will be used later for classifying each track to one of 5 genres (rock, pop, poprock, rap, and others).



```python
def create_song_df(track_features=list, artist_features=list, pid=list):

    """
    This function takes in two lists of track and artist features, respectively,
    and generates a dataframe of the features.

    inputs:
        1. track_features: list of all tracks including features
        2. artist_features: list of all artists including features

    returns:
        1. df: a pandas dataframe of size (N, X) where N corresponds to the number of songs
        in track_features, X is the number of features in the dataframe.
    """

    import pandas as pd

    selected_song_features = ['uri', 'duration_ms', 'time_signature', 'key',
                              'tempo', 'energy', 'mode', 'loudness', 'speechiness',
                              'danceability', 'acousticness', 'instrumentalness',
                              'valence', 'liveness']
    selected_artist_features = ['followers', 'uri', 'name', 'popularity', 'genres']

    col_names = ['song_uri', 'duration_ms', 'time_signature', 'key',
                 'tempo', 'energy', 'mode', 'loudness', 'speechiness',
                 'danceability', 'acousticness', 'instrumentalness',
                 'valence', 'liveness', 'artist_followers', 'artist_uri',
                 'artist_name', 'artist_popularity']


    data = []

    for i, j in zip(track_features, artist_features):
        temp = []
        for sf in selected_song_features:
            temp.append(i.get(sf))
        for af in selected_artist_features:
            if af == 'followers':
                temp.append(j.get('followers').get('total'))
            elif af == 'genres':
                for g in j.get('genres'):
                    temp.append(g)
            else:
                temp.append(j.get(af))

        data.append(list(temp))

    df = pd.DataFrame(data)

    for i in range(len(df.columns)- len(col_names)):
        col_names.append('g'+str(i+1))

    df.columns = col_names

    df.insert(loc=0, column='pid', value=pid)

    return df
```




```python
songs_df = create_song_df(t_features, a_features, pid_t)
songs_df.head()
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
      <th>pid</th>
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
      <th>g1</th>
      <th>g2</th>
      <th>g3</th>
      <th>g4</th>
      <th>g5</th>
      <th>g6</th>
      <th>g7</th>
      <th>g8</th>
      <th>g9</th>
      <th>g10</th>
      <th>g11</th>
      <th>g12</th>
      <th>g13</th>
      <th>g14</th>
      <th>g15</th>
      <th>g16</th>
      <th>g17</th>
      <th>g18</th>
      <th>g19</th>
      <th>g20</th>
      <th>g21</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
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
      <td>909647</td>
      <td>spotify:artist:2wIVse2owClT7go1WT98tk</td>
      <td>Missy Elliott</td>
      <td>76</td>
      <td>dance pop</td>
      <td>hip hop</td>
      <td>hip pop</td>
      <td>pop</td>
      <td>pop rap</td>
      <td>r&amp;b</td>
      <td>rap</td>
      <td>southern hip hop</td>
      <td>urban contemporary</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
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
      <td>5457673</td>
      <td>spotify:artist:26dSoYclwsYLMAKD3tpOr4</td>
      <td>Britney Spears</td>
      <td>82</td>
      <td>dance pop</td>
      <td>pop</td>
      <td>post-teen pop</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
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
      <td>16686181</td>
      <td>spotify:artist:6vWDO969PvNqNYHIOW5v0m</td>
      <td>Beyoncé</td>
      <td>87</td>
      <td>dance pop</td>
      <td>pop</td>
      <td>post-teen pop</td>
      <td>r&amp;b</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
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
      <td>7343717</td>
      <td>spotify:artist:31TPClRtHm23RisEBtV3X7</td>
      <td>Justin Timberlake</td>
      <td>83</td>
      <td>dance pop</td>
      <td>pop</td>
      <td>pop rap</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
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
      <td>1044930</td>
      <td>spotify:artist:5EvFsr3kj42KNv97ZEnqij</td>
      <td>Shaggy</td>
      <td>74</td>
      <td>dance pop</td>
      <td>pop rap</td>
      <td>reggae fusion</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## Collapsing songs to unique playlists

This section is responsible for collapsing songs to unique playlist IDs, so that for each playlist we would have a vector of average of the features of songs belonging to a playlist, which characterizes each playlist. In this section we also classified songs, and playlists.

The following function classifies songs according to the given genres of the artist of the song, according to "if" statements:



```python
def genre_generator(songs_df):

    """
    This function classifies songs according to the given genres of the artist of the song, according to an "if" statements.

    Input: dataframe with a list of songs

    Output: dataframe with added column with unique genre for each song

    """
    # defining liist of genres that will determine a song with unique genre "rap"
    rap = ["rap","hiphop", "r&d"]

    # finding position of "g1" (first column of genres) and last position of "gX" in columns (last column of genres) , to use it later for assessingn genre of song
    g1_index = 0
    last_column_index = 0

    column_names = songs_df.columns.values

    # finding first column with genres ("g1")
    for i in column_names:
        if i == "g1":
            break
        g1_index += 1

    # finding last column with genrer ("gX")
    for i in column_names:
        last_column_index += 1

    # create new columnn that will have unique genre (class) of each song
    songs_df["genre"] = ""

    # loop to create genre for each song in dataframe     
    for j in range(len(songs_df)):

        # Creating list of genres for a given song  
        genres_row = list(songs_df.iloc[[j]][column_names[g1_index:last_column_index-1]].dropna(axis=1).values.flatten())
        # genres_row = ['british invasion', 'merseybeat', 'psychedelic']

        # classifing genre for the song

        genre = "other"

        if any("rock" in s for s in genres_row) and any("pop" in s for s in genres_row):
            genre = "pop rock"
        elif any("rock" in s for s in genres_row):
            genre = "rock"
        elif any("pop" in s for s in genres_row):
            genre = "pop"

        for i in rap:
            if any(i in s for s in genres_row):
                genre = "rap"

        # giving column genre the classified genre for a given song         
        songs_df.set_value(j, 'genre', genre)

    return songs_df
```


The code below calls the song genre generator function, and the result is a dataframe with songs containing a genre, which has been classified according to the genre of the artists of each song.



```python
songs_df_new = genre_generator(songs_df)
songs_df_new.head()
```


    C:\Users\Joao Araujo\Anaconda3\lib\site-packages\ipykernel_launcher.py:56: FutureWarning: set_value is deprecated and will be removed in a future release. Please use .at[] or .iat[] accessors instead





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
      <th>pid</th>
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
      <th>g1</th>
      <th>g2</th>
      <th>g3</th>
      <th>g4</th>
      <th>g5</th>
      <th>g6</th>
      <th>g7</th>
      <th>g8</th>
      <th>g9</th>
      <th>g10</th>
      <th>g11</th>
      <th>g12</th>
      <th>g13</th>
      <th>g14</th>
      <th>g15</th>
      <th>g16</th>
      <th>g17</th>
      <th>g18</th>
      <th>g19</th>
      <th>g20</th>
      <th>g21</th>
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
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
      <td>909647</td>
      <td>spotify:artist:2wIVse2owClT7go1WT98tk</td>
      <td>Missy Elliott</td>
      <td>76</td>
      <td>dance pop</td>
      <td>hip hop</td>
      <td>hip pop</td>
      <td>pop</td>
      <td>pop rap</td>
      <td>r&amp;b</td>
      <td>rap</td>
      <td>southern hip hop</td>
      <td>urban contemporary</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
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
      <td>5457673</td>
      <td>spotify:artist:26dSoYclwsYLMAKD3tpOr4</td>
      <td>Britney Spears</td>
      <td>82</td>
      <td>dance pop</td>
      <td>pop</td>
      <td>post-teen pop</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
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
      <td>16686181</td>
      <td>spotify:artist:6vWDO969PvNqNYHIOW5v0m</td>
      <td>Beyoncé</td>
      <td>87</td>
      <td>dance pop</td>
      <td>pop</td>
      <td>post-teen pop</td>
      <td>r&amp;b</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
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
      <td>7343717</td>
      <td>spotify:artist:31TPClRtHm23RisEBtV3X7</td>
      <td>Justin Timberlake</td>
      <td>83</td>
      <td>dance pop</td>
      <td>pop</td>
      <td>pop rap</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
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
      <td>1044930</td>
      <td>spotify:artist:5EvFsr3kj42KNv97ZEnqij</td>
      <td>Shaggy</td>
      <td>74</td>
      <td>dance pop</td>
      <td>pop rap</td>
      <td>reggae fusion</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>None</td>
      <td>rap</td>
    </tr>
  </tbody>
</table>
</div>



The following lines clean the dataframe by dropping unnecessary columns (the genres of each song), which were used to create the unique column of song genre that will be used later in the algorithm.



```python
temp = songs_df_new.copy()
```




```python
column_names_temp = songs_df_new.columns.values[18:-1]
column_names_temp
```





    array(['artist_popularity', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8',
           'g9', 'g10', 'g11', 'g12', 'g13', 'g14', 'g15', 'g16', 'g17', 'g18',
           'g19', 'g20', 'g21'], dtype=object)





```python
temp = temp.drop(column_names_temp,axis=1)
temp.head()
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
      <th>pid</th>
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
      <th>genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
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
      <td>909647</td>
      <td>spotify:artist:2wIVse2owClT7go1WT98tk</td>
      <td>Missy Elliott</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
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
      <td>5457673</td>
      <td>spotify:artist:26dSoYclwsYLMAKD3tpOr4</td>
      <td>Britney Spears</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
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
      <td>16686181</td>
      <td>spotify:artist:6vWDO969PvNqNYHIOW5v0m</td>
      <td>Beyoncé</td>
      <td>pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
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
      <td>7343717</td>
      <td>spotify:artist:31TPClRtHm23RisEBtV3X7</td>
      <td>Justin Timberlake</td>
      <td>rap</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
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
      <td>1044930</td>
      <td>spotify:artist:5EvFsr3kj42KNv97ZEnqij</td>
      <td>Shaggy</td>
      <td>rap</td>
    </tr>
  </tbody>
</table>
</div>





```python
feature_indexes = list(range(len(temp.columns)-1))
```




```python
col_names_temp = ['duration_ms','time_signature','key','tempo','energy','loudness','speechiness','danceability','acousticness',
         'instrumentalness', 'valence', 'liveness', 'artist_followers', 'artist_popularity'  ]

```




```python
col_names = temp.columns
```


The code below one-hot-encodes the variable genre, so that we can calculated the proportion of songs of each genre in each playlist. This will help classify the genre of our playlist according to the most frequent genre of songs that belong to that playlist.



```python
songs_encoded = pd.get_dummies(temp,columns = ['genre'],drop_first=False)
songs_encoded.head()
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
      <th>pid</th>
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
      <th>genre_other</th>
      <th>genre_pop</th>
      <th>genre_pop rock</th>
      <th>genre_rap</th>
      <th>genre_rock</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
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
      <td>909647</td>
      <td>spotify:artist:2wIVse2owClT7go1WT98tk</td>
      <td>Missy Elliott</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
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
      <td>5457673</td>
      <td>spotify:artist:26dSoYclwsYLMAKD3tpOr4</td>
      <td>Britney Spears</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
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
      <td>16686181</td>
      <td>spotify:artist:6vWDO969PvNqNYHIOW5v0m</td>
      <td>Beyoncé</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
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
      <td>7343717</td>
      <td>spotify:artist:31TPClRtHm23RisEBtV3X7</td>
      <td>Justin Timberlake</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
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
      <td>1044930</td>
      <td>spotify:artist:5EvFsr3kj42KNv97ZEnqij</td>
      <td>Shaggy</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The following function takes a data frame of songs (with playlists IDs) and collapses the dataframe at the playlist ID level, to get averages for each column (which characterize each playlist). This creates a datafram at the playlist level.



```python
def collapse_pid(df):

    """
    This function takes a data frame of songs (with playlists IDs) and collapses the dataframe at the playlist ID level, to get averages for each column.

    Input: data frame of songs (with playlists IDs)

    Output: data frame of playlists (collapsing songs into playlist IDs, using average)

    """

    # Group by play list category
    pid_groups = df.groupby('pid')
    # Apply mean function to all columns

    return pid_groups.mean()

playlists_collapsed = collapse_pid(songs_encoded)

```


## Classifying each playlist to one of 5 genres (rock, pop, poprock, rap, and others)

The following function classifies playlists according to the most frequent genre of the songs in the playlist:



```python
def playlist_genre_generator (df, first_row):

    """
    This function classifies playlists according to the most frequent genre of the songs in the playlist

    Input: dataframe with a list of playlists

    Output: dataframe with added column with unique genre for each playlist

    """

    # create new columnn that will have unique genre (class) of each playlist
    df ["playlist_genre"] = ""

    for j in range(len(df)):

        # finding position of "g1" (first column of genres) and last position of "gX" in columns (last column of genres) , to use it later for assessingn genre of song
        g1_index = 0
        last_column_index = 0

        column_names = df.columns.values

        # finding first column with genres ("g1")
        for i in column_names:
            if i == "artist_followers":
                break
            g1_index += 1
        g1_index += 1

        # finding last column with genrer ("gX")
        for i in column_names:
            last_column_index += 1
        last_column_index -= 1

        # Creating list of genres for a given song  
        genres_row = list(df.iloc[[j]][column_names[g1_index:last_column_index]].dropna(axis=1).values.flatten())

        # classifing genre for the playlist
        max_value = max(genres_row)
        max_index = genres_row.index(max_value)
        playlist_genre = column_names[g1_index + max_index]

        # giving column genre the classified genre for a given playlist
        df.set_value(j + first_row, 'playlist_genre', playlist_genre)
    return df
```


## Setting up final sample of playlists of equally distributed in each of the classes (genres)

The following code creates a "base line" playlist with a defined minimum size of the playlist (2000 playlists), which will have an unequal distribution of genres among the playlists, as demonstrated in the output table below.



```python
### creating base_line data frame

import warnings
warnings.filterwarnings('ignore')

n_playlist = 2000
pid_t, track_uri = feature_list_func(plylist, feature = 'track_uri', n_playlist = n_playlist, first_pid = 0)
pid_a, artist_uri = feature_list_func(plylist, feature = 'artist_uri', n_playlist = n_playlist, first_pid = 0)

t_features, a_features = get_all_features(track_uri, artist_uri, sp)

#create dataframe of songs
songs_df = create_song_df(t_features, a_features, pid_t)
songs_df_new = genre_generator(songs_df)
temp = songs_df_new.copy()
column_names_temp = songs_df_new.columns.values[18:-1]
temp = temp.drop(column_names_temp,axis=1)
songs_encoded = pd.get_dummies(temp,columns = ['genre'],drop_first=False)

#create dataframe of playlists
playlists_collapsed = collapse_pid(songs_encoded)
genre_classified_playlists = playlist_genre_generator (playlists_collapsed, first_row = 0)
genre_classified_playlists.head()
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
      <th>genre_other</th>
      <th>genre_pop</th>
      <th>genre_pop rock</th>
      <th>genre_rap</th>
      <th>genre_rock</th>
      <th>playlist_genre</th>
    </tr>
    <tr>
      <th>pid</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>221777.461538</td>
      <td>4.000000</td>
      <td>5.038462</td>
      <td>123.006885</td>
      <td>0.782173</td>
      <td>0.692308</td>
      <td>-4.881942</td>
      <td>0.107021</td>
      <td>0.659288</td>
      <td>0.083440</td>
      <td>0.000676</td>
      <td>0.642904</td>
      <td>0.192127</td>
      <td>4.800843e+06</td>
      <td>0.000000</td>
      <td>0.288462</td>
      <td>0.230769</td>
      <td>0.461538</td>
      <td>0.019231</td>
      <td>genre_rap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>298844.128205</td>
      <td>3.769231</td>
      <td>4.461538</td>
      <td>122.669615</td>
      <td>0.691077</td>
      <td>0.538462</td>
      <td>-8.291667</td>
      <td>0.088449</td>
      <td>0.496459</td>
      <td>0.163100</td>
      <td>0.222270</td>
      <td>0.476667</td>
      <td>0.178433</td>
      <td>1.704673e+06</td>
      <td>0.358974</td>
      <td>0.000000</td>
      <td>0.051282</td>
      <td>0.000000</td>
      <td>0.589744</td>
      <td>genre_rock</td>
    </tr>
    <tr>
      <th>2</th>
      <td>219374.875000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>114.600672</td>
      <td>0.693203</td>
      <td>0.515625</td>
      <td>-4.874156</td>
      <td>0.096288</td>
      <td>0.671875</td>
      <td>0.269230</td>
      <td>0.000638</td>
      <td>0.565078</td>
      <td>0.169028</td>
      <td>1.691574e+06</td>
      <td>0.062500</td>
      <td>0.937500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>genre_pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>229575.055556</td>
      <td>3.952381</td>
      <td>5.103175</td>
      <td>125.032413</td>
      <td>0.621282</td>
      <td>0.714286</td>
      <td>-9.614937</td>
      <td>0.067186</td>
      <td>0.513714</td>
      <td>0.273870</td>
      <td>0.202042</td>
      <td>0.451623</td>
      <td>0.188585</td>
      <td>2.125109e+05</td>
      <td>0.246032</td>
      <td>0.150794</td>
      <td>0.317460</td>
      <td>0.071429</td>
      <td>0.214286</td>
      <td>genre_pop rock</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255014.352941</td>
      <td>3.941176</td>
      <td>3.352941</td>
      <td>127.759882</td>
      <td>0.650535</td>
      <td>0.823529</td>
      <td>-7.634471</td>
      <td>0.041159</td>
      <td>0.576765</td>
      <td>0.177148</td>
      <td>0.081875</td>
      <td>0.490765</td>
      <td>0.166524</td>
      <td>1.167521e+06</td>
      <td>0.117647</td>
      <td>0.117647</td>
      <td>0.705882</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>genre_pop rock</td>
    </tr>
  </tbody>
</table>
</div>



The following code is an intermediary step in adjusting the sample towards an equal distribution of genres among all playlists. It looks for the most frequent genre among the playlists, calculates the number of playlists of each genre, so that in the next step we fill up the sample with playlits of underrepresented genres.  



```python
from pandas.tools.plotting import table
table = genre_classified_playlists['playlist_genre'].value_counts()
mode_genre = genre_classified_playlists['playlist_genre'].value_counts().idxmax()
number_mode_genre = table.loc[mode_genre]

number_genre_pop = table.loc["genre_pop"]
number_genre_rap = table.loc["genre_rap"]
number_genre_other = table.loc["genre_other"]
number_genre_poprock = table.loc["genre_pop rock"]
number_genre_rock = table.loc["genre_rock"]

mode_genre = genre_classified_playlists['playlist_genre'].value_counts().idxmax()
mode_genre
total_number = number_genre_pop + number_genre_rap + number_genre_other + number_genre_poprock + number_genre_rock
total_number
```





    2675



The code below takes one playlist at a time from the pool of 15,000 playlists (read from the Million Playlist json files at the beginning of this page), checks to which genre it belongs, and adds the playlist (if of underepresented genre) to the baseline sample, until the full sample is equally distributed.

The playlists taken from the 15,000 playlists are taken in sequence after the playlists that have already been added to the sample, or discarded if the playlist belongs to an already "well represented genre".



```python
### adjusting base_line data frame to get to desired distribution

start_time = time.time()

t = 0

while total_number < number_mode_genre*5:

    first_pid = n_playlist + t

    # get uri for tracks and artists of playlist selected
    pid_t, track_uri = feature_list_func(plylist, feature = 'track_uri', n_playlist = 1, first_pid = first_pid)
    pid_a, artist_uri = feature_list_func(plylist, feature = 'artist_uri', n_playlist = 1, first_pid = first_pid)
    t_features, a_features = get_all_features(track_uri, artist_uri, sp)

    #create dataframe of songs
    songs_df = create_song_df(t_features, a_features, pid_t)
    songs_df_new = genre_generator(songs_df)
    temp = songs_df_new.copy()
    column_names_temp = songs_df_new.columns.values[18:-1]
    temp = temp.drop(column_names_temp,axis=1)
    temp

    songs_encoded = pd.get_dummies(temp,columns = ['genre'],drop_first=False)
    songs_encoded

    #create dataframe of playlists
    playlists_collapsed = collapse_pid(songs_encoded)
    genre_classified_SinglePlaylist = playlist_genre_generator (playlists_collapsed, first_row = first_pid)

    # checking if playlist selected belongs to one of the genres that is not the most frequent in baseline dataframe
    if total_number != 5*number_mode_genre:

        if genre_classified_SinglePlaylist.playlist_genre.iloc[0] == "genre_pop":
            if number_genre_pop < number_mode_genre:
                genre_classified_playlists = genre_classified_playlists.append(genre_classified_SinglePlaylist, sort=False)
                number_genre_pop += 1
        elif genre_classified_SinglePlaylist.playlist_genre.iloc[0] == "genre_rap":
            if number_genre_rap < number_mode_genre:
                genre_classified_playlists = genre_classified_playlists.append(genre_classified_SinglePlaylist, sort=False)
                number_genre_rap += 1
        elif genre_classified_SinglePlaylist.playlist_genre.iloc[0] == "genre_other":
            if number_genre_other < number_mode_genre:
                genre_classified_playlists = genre_classified_playlists.append(genre_classified_SinglePlaylist, sort=False)
                number_genre_other += 1
        elif genre_classified_SinglePlaylist.playlist_genre.iloc[0] == "genre_pop rock":
            if number_genre_poprock < number_mode_genre:
                genre_classified_playlists = genre_classified_playlists.append(genre_classified_SinglePlaylist, sort=False)
                number_genre_poprock += 1
        elif genre_classified_SinglePlaylist.playlist_genre.iloc[0] == "genre_rock":
            if number_genre_rock < number_mode_genre:
                genre_classified_playlists = genre_classified_playlists.append(genre_classified_SinglePlaylist, sort=False)
                number_genre_rock += 1

    t += 1

    total_number = number_genre_pop + number_genre_rap + number_genre_other + number_genre_poprock + number_genre_rock

    # print (total_number)
    # print (number_genre_pop)
    # print (number_genre_rap)
    # print (number_genre_other)
    # print (number_genre_poprock)
    # print (number_genre_rock)

print("--- %s seconds ---" % (time.time() - start_time))

genre_classified_playlists.head()
```


    --- 0.0009975433349609375 seconds ---





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
      <th>pid</th>
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
      <th>genre_other</th>
      <th>genre_pop</th>
      <th>genre_pop rock</th>
      <th>genre_rap</th>
      <th>genre_rock</th>
      <th>playlist_genre</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>221777.461538</td>
      <td>4.000000</td>
      <td>5.038462</td>
      <td>123.006885</td>
      <td>0.782173</td>
      <td>0.692308</td>
      <td>-4.881942</td>
      <td>0.107021</td>
      <td>0.659288</td>
      <td>0.083440</td>
      <td>0.000676</td>
      <td>0.642904</td>
      <td>0.192127</td>
      <td>4.797984e+06</td>
      <td>0.000000</td>
      <td>0.288462</td>
      <td>0.230769</td>
      <td>0.461538</td>
      <td>0.019231</td>
      <td>genre_rap</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>298844.128205</td>
      <td>3.769231</td>
      <td>4.461538</td>
      <td>122.669615</td>
      <td>0.691077</td>
      <td>0.538462</td>
      <td>-8.291667</td>
      <td>0.088449</td>
      <td>0.496459</td>
      <td>0.163100</td>
      <td>0.222270</td>
      <td>0.476667</td>
      <td>0.178433</td>
      <td>1.702573e+06</td>
      <td>0.358974</td>
      <td>0.000000</td>
      <td>0.051282</td>
      <td>0.000000</td>
      <td>0.589744</td>
      <td>genre_rock</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>219374.875000</td>
      <td>4.000000</td>
      <td>5.000000</td>
      <td>114.600672</td>
      <td>0.693203</td>
      <td>0.515625</td>
      <td>-4.874156</td>
      <td>0.096288</td>
      <td>0.671875</td>
      <td>0.269230</td>
      <td>0.000638</td>
      <td>0.565078</td>
      <td>0.169028</td>
      <td>1.688725e+06</td>
      <td>0.062500</td>
      <td>0.937500</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>genre_pop</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>229575.055556</td>
      <td>3.952381</td>
      <td>5.103175</td>
      <td>125.032413</td>
      <td>0.621282</td>
      <td>0.714286</td>
      <td>-9.614937</td>
      <td>0.067186</td>
      <td>0.513714</td>
      <td>0.273870</td>
      <td>0.202042</td>
      <td>0.451623</td>
      <td>0.188585</td>
      <td>2.123258e+05</td>
      <td>0.246032</td>
      <td>0.150794</td>
      <td>0.317460</td>
      <td>0.071429</td>
      <td>0.214286</td>
      <td>genre_pop rock</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>255014.352941</td>
      <td>3.941176</td>
      <td>3.352941</td>
      <td>127.759882</td>
      <td>0.650535</td>
      <td>0.823529</td>
      <td>-7.634471</td>
      <td>0.041159</td>
      <td>0.576765</td>
      <td>0.177148</td>
      <td>0.081875</td>
      <td>0.490765</td>
      <td>0.166524</td>
      <td>1.166320e+06</td>
      <td>0.117647</td>
      <td>0.117647</td>
      <td>0.705882</td>
      <td>0.000000</td>
      <td>0.058824</td>
      <td>genre_pop rock</td>
    </tr>
  </tbody>
</table>
</div>



Finally, we check to make sure that the final dataframe is equally distributed among all genres:



```python
display(genre_classified_playlists['playlist_genre'].value_counts())
display(genre_classified_playlists['playlist_genre'].value_counts(normalize=True))
```



    genre_other       535
    genre_pop         535
    genre_rock        535
    genre_rap         535
    genre_pop rock    535
    Name: playlist_genre, dtype: int64



    genre_other       0.2
    genre_pop         0.2
    genre_rock        0.2
    genre_rap         0.2
    genre_pop rock    0.2
    Name: playlist_genre, dtype: float64


And export the final dataframe as a csv file, which will be used as the sample data for our machine learning models. This sample will be split into training and test data, the former for training different models and assesing their performance, and the latter for evaluating how well our trained models perform in the test data.



```python
genre_classified_playlists.to_csv ("playlist_df.csv")
```
