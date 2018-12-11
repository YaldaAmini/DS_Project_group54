
# coding: utf-8

# In[ ]:


def feature_list_func(plylist_dic, feature, n_playlist):
    """"
    This function takes a number of playlist and returns the features of the tracks of those selected playlists.

    input:
        1 - plylist_dic: dictionary of all playlists (dataset in dictionary format: json)
        2 - feature: feature to be selected from each songs in selected playlists
        3 - n_playlists: number of playlists to be selected

    output: list of observations for the feature chosen, for all of the tracks that belong to the selected playlists

    """
    import numpy as np

    feature_list = []
    pid_list = []
    length_playlist = np.minimum(n_playlist,len(plylist_dic.get('playlists'))) # the output will be based on the min of the n_playlist and the actual length of the input playlist
    for i in range(length_playlist):
        playlist = plylist_dic.get('playlists')[i]

        playlist_pid = playlist.get('pid')
        for j in range(len(playlist.get('tracks'))):
            feature_list.append(playlist.get('tracks')[j].get(feature))
            pid_list.append(playlist_pid)
    return pid_list, feature_list


# In[ ]:


def create_spotipy_obj():
    """
    Uses dbarjum's client id for DS Project
    """
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    import spotipy.util as util
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


# In[ ]:


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


def genre_generator(songs_df):

    # Defining genres to single genre
    rap = ["rap","hiphop", "r&d"]

    # finding position of g1 and last position of gX in columns, to use it later for assessingn genre of song

    g1_index = 0
    last_column_index = 0

    column_names = songs_df.columns.values

    for i in column_names:
        if i == "g1":
            break
        g1_index += 1

    for i in column_names:
        last_column_index += 1

    # loop to create gender for each song in dataframe

    songs_df["genre"] = ""

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

def plot_dist_features(df, features, save=False, path=str):

    """
    Plots the distribution of all features passed into the function and saves them
    if save == True.

    Arguments:
        - df: the dataframe used to plot features
        - features: the column names of the dataframe whose features are to be plot
        - save: boolean indicating whether the plots are to be saved
        - path: string indicating the path of where to save the features.

    Return: None, just prints the distributions
    """

    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import seaborn as sns

    for i in features:
        fig, axs = plt.subplots(1, figsize=(9,6))
        sns.set(style='white', palette='deep')

        x = df[i]
        if i == 'duration_ms':
            x = x/1000
            sns.kdeplot(x, label = 'duration in seconds', shade=True).set_title("Distribution of Feature Duration")
        else:
            sns.kdeplot(x, label = i, shade=True).set_title("Distribution of Feature "+i)

        if save:
            filename='distribution_plot_'+i
            fig.savefig(path+filename)

    return

def get_playlist_n(playlist_dic, feature, n_playlist):
    """"
    This function takes a number of playlist and returns the features of the tracks of those selected playlists.

    input:
        1 - plylist_dic: dictionary of all playlists (dataset in dictionary format: json)
        2 - feature: feature to be selected from each songs in selected playlists
        3 - n_playlists: number of playlists to be selected

    output: list of observations for the feature chosen, for all of the tracks that belong to the selected playlists

    """
    import numpy as np

    feature_list = []

    for j in range(len(playlist_dic.get('tracks'))):
        feature_list.append(playlist_dic.get('tracks')[j].get(feature))

    return feature_list
