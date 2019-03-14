import warnings
import urllib
import lxml.html
import pandas as pd
from textblob import TextBlob
from sklearn.neighbors import NearestNeighbors
import numpy as np

warnings.simplefilter("ignore")
metadata=pd.read_csv("data/songs.csv")
metadata=metadata.tail(10)
class Song(object):
    def __init__(self, artist, title):
        self.artist = self.__format_str(artist)
        self.title = self.__format_str(title)
        self.url = None
        self.lyric = None
        
    def __format_str(self, s):
        try:
            s = s.strip()
        
            s = ''.join(c for c in unicodedata.normalize('NFD', s)
                         if unicodedata.category(c) != 'Mn')
        except:
            pass
        s = s.title()
        return s
        
    def __quote(self, s):
         return urllib.parse.quote(s.replace(' ', '_'))

    def __make_url(self):
        artist = self.__quote(self.artist)
        title = self.__quote(self.title)
        artist_title = '%s:%s' %(artist, title)
        url = 'http://lyrics.wikia.com/' + artist_title
        self.url = url
        
    def update(self, artist=None, title=None):
        if artist:
            self.artist = self.__format_str(artist)
        if title:
            self.title = self.__format_str(title)
        
    def lyricwikia(self):
        self.__make_url()
        try:
            doc = lxml.html.parse(self.url)
            lyricbox = doc.getroot().cssselect('.lyricbox')[0]
        except (IOError, IndexError) as e:
            self.lyric = ''
            return self.lyric
        lyrics = []

        for node in lyricbox:
            if node.tag == 'br':
                lyrics.append('\n')
            if node.tail is not None:
                lyrics.append(node.tail)
        self.lyric =  "".join(lyrics).strip()    
        return self.lyric
for index,row in metadata.iterrows():
    song = Song(artist=row['artist_name'], title=row['title'])
    lyr = song.lyricwikia()
    metadata.at[index,'lyrics']=lyr
    polarity=TextBlob(lyr).sentiment[0]
    metadata.at[index,'polarity_score']=polarity
    
lyrics_data=metadata
lyrics_data=lyrics_data[lyrics_data['lyrics']!='']
lyrics_data = lyrics_data.reset_index()
df=lyrics_data
for index,row in df.iterrows():
    if row['polarity_score']>0:
        df.at[index,'happy']=1
    else:
        df.at[index,'happy']=0
        
polarity=df['polarity_score']
happy=df['happy']
df=pd.concat([polarity,happy],axis=1)
df=np.array(df)


#nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(X)
#print(nbrs.predict([0.4]))
#neigh = NearestNeighbors(n_neighbors=10)
#neigh.fit(X)
#NearestNeighbors(algorithm='ball_tree', leaf_size=100,metric='euclidean')
nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(df)
distances, indices = nbrs.kneighbors([[0.26453636,1]])
print(lyrics_data.get_value(indices[0][0],'title'))