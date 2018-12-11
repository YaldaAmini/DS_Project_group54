def suggest_genre_songs(N: int, genre: str, playlist, pool_path: str):
    """
    This function recommends N best songs to be added toa playlist based on
    the genre of the playlist.
    INPUTS:
        N: (int) number of songs to return as recommendations
        genre: (string) classification based on our model
        playlist: (pandas dataframe) a pandas dataframe containing the playlist
            being analyzed.
        pool_path: (string) path to the pool file
    OUTPUT:
        df: (pandas df) df of recommended songs with some information
    """
    import pandas as pd
    import numpy as np
    import spotify_api_function_set as spf

    dtypes = {'song_uri': str, 'duration_ms': np.int64, 'time_signature': np.int64,
              'key': np.int64, 'tempo': np.float64, 'energy': np.float64, 'mode': np.int64,
              'loudness': np.float64, 'speechiness': np.float64, 'danceability': np.float64,
              'acousticness': np.float64,'instrumentalness': np.float64, 'valence': np.float64,
              'liveness': np.float64, 'artist_followers': np.int64, 'artist_uri': str,
              'artist_name': str, 'artist_popularity': np.int64, 'genre': str}

    pool = pd.read_csv(pool_path, dtype=dtypes).sample(frac=1)
    pool = pool.loc[pool['genre'] == genre]

    songs = []
    i = 0
    while len(songs) < N:
        song = pool.song_uri.iloc[i]
        if not song in playlist:
            songs.append(song)

        i += 1

    sp = spf.create_spotipy_obj()
    tracks =  sp.tracks(songs)

    df_data = []
    for i in tracks['tracks']:
        song_name = i['name']
        artist_name = i['artists'][0]['name']
        album_name = i['album']['name']
        song_uri = i['uri']
        df_data.append([song_name, artist_name, album_name, song_uri])

    df = pd.DataFrame(df_data)
    df.columns = ['song_name', 'artist_name', 'album_name', 'song_uri']

    return df

def suggest_best_songs(feature_vector, N: int, genre: str, playlist, pool_path: str):
    """
    This function recommends N best songs to be added toa playlist based on
    cosine similarity between the suggested songs and the songs in the playlist.
    INPUTS:
        N: (int) number of songs to return as recommendations
        genre: (string) classification based on our model
        feature_vector: (np series) a vector of audio features to use for cosine
            similarity comparison.
        pool_path: (string) path to the pool file
    OUTPUT:
        df: (pandas df) df of recommended songs with some information
    """
    import pandas as pd
    import numpy as np
    import spotify_api_function_set as spf
    from sklearn.metrics.pairwise import cosine_similarity

    dtypes = {'song_uri': str, 'duration_ms': np.int64, 'time_signature': np.int64,
              'key': np.int64, 'tempo': np.float64, 'energy': np.float64, 'mode': np.int64,
              'loudness': np.float64, 'speechiness': np.float64, 'danceability': np.float64,
              'acousticness': np.float64,'instrumentalness': np.float64, 'valence': np.float64,
              'liveness': np.float64, 'artist_followers': np.int64, 'artist_uri': str,
              'artist_name': str, 'artist_popularity': np.int64, 'genre': str}

    pool = pd.read_csv(pool_path, dtype=dtypes)
    pool = pool.loc[pool['genre'] == genre]
    song_uri = pool.song_uri
    drop = set(pool.columns)^set(list(feature_vector.index))

    pool = pool.drop(drop, axis=1)

    feature_vector = np.array(feature_vector.values.reshape(1,-1))

    similarity = []
    for index, row in pool.iterrows():
        similarity.append(float(cosine_similarity(np.array(row.values.reshape(1,-1)),feature_vector)))

    pool['similarity'] = similarity
    pool['song_uri'] = song_uri
    pool = pool.sort_values(by=['similarity'], ascending=False)

    songs = []
    i = 0
    while len(songs) < N:
        song = pool.song_uri.iloc[i]
        if not song in playlist:
            songs.append(song)

        i += 1

    sp = spf.create_spotipy_obj()
    tracks =  sp.tracks(songs)

    df_data = []
    for i in tracks['tracks']:
        song_name = i['name']
        artist_name = i['artists'][0]['name']
        album_name = i['album']['name']
        song_uri = i['uri']
        df_data.append([song_name, artist_name, album_name, song_uri])

    df = pd.DataFrame(df_data)
    df.columns = ['song_name', 'artist_name', 'album_name', 'song_uri']

    return df
