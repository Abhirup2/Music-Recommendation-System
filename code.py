import pandas as pd
df=pd.read_csv("/content/spotify_millsongdata.csv")

df=df.sample(5000).drop('link',axis=1).reset_index(drop=True)
df.head(10)
df['text']=df['text'].str.lower().replace(r'^\W\s',' ').replace(r'\n',' ', regex=True)

nltk.download('all')

import nltk
from nltk.stem  import PorterStemmer
stemmer=PorterStemmer()

def token(text):
    token=nltk.word_tokenize(text)
    a=[stemmer.stem(w) for w in token]
    return " ".join(a)
df['text'].apply(lambda x: token(x))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf_vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')

tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

similar_content = cosine_similarity(tfidf_matrix)

pip install scikit-surprise
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split

df['rating'] = 1.0

reader = Reader(rating_scale=(0, 1))

data = Dataset.load_from_df(df[['artist', 'song', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based collaborative filtering
}

model = KNNBasic(sim_options=sim_options)
model.fit(trainset)

def content_based_recommender(song_name):
    idx = df[df['song'] == song_name].index[0]
    distance = sorted(list(enumerate(similar_content[idx])), reverse=True, key=lambda x: x[1])
    recommended_songs_content = [df.iloc[s[0]]['song'] for s in distance[1:23]]
    return recommended_songs_content

def collaborative_recommender(song_name):
    song_id = df[df['song'] == song_name].index[0]
    similar_items = model.get_neighbors(song_id, k=10)
    recommended_songs_collab = [df.iloc[song]['song'] for song in similar_items]
    return recommended_songs_collab


def hybrid_recommender(song_name):
    content_recommendations = content_based_recommender(song_name)
    collaborative_recommendations = collaborative_recommender(song_name)

    # Combine recommendations, giving more weight to content-based recommendations
    hybrid_recommendations = content_recommendations + collaborative_recommendations[:10]

    return hybrid_recommendations

song_to_recommend = 'Paradise'
 

content_based_recommendations = content_based_recommender(song_to_recommend)
collaborative_recommendations = collaborative_recommender(song_to_recommend)
hybrid_recommendations=hybrid_recommender(song_to_recommend)

    # Displaying recommendations
print(f"Content-Based Recommendations for '{song_to_recommend}': {content_based_recommendations}")
print(f"Collaborative Filtering Recommendations for '{song_to_recommend}': {collaborative_recommendations}")
print(f"Hybrid Filtering Recommendations for '{song_to_recommend}': {hybrid_recommendations}")






























