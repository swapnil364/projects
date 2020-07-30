import streamlit as st
import time
from PIL import Image
from tmdbv3api import TMDb
from tmdbv3api import Movie
from tmdbv3api import Person
import requests
import pandas as pd
import re
from bs4 import BeautifulSoup as bs
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


@st.cache(show_spinner=False)
def extract_features(movies, main):

	genre_vector = CountVectorizer().fit_transform(movies.genre)
	director_vector = CountVectorizer(max_features=500).fit_transform(movies.director)
	casts_vector = CountVectorizer(max_features=5000).fit_transform(movies.casts)
	genre_vector_data = pd.DataFrame(genre_vector.toarray(), columns = ['genre'+str(i) for i in range(genre_vector.shape[1])])
	director_vector_data = pd.DataFrame(director_vector.toarray(), columns = ['director'+str(i) for i in range(director_vector.shape[1])])
	casts_vector_data = pd.DataFrame(casts_vector.toarray(), columns = ['casts'+str(i) for i in range(casts_vector.shape[1])])
	nums = movies[['release_date', 'popularity', 'vote_count', 'imdb_rating']]
	vectors = pd.concat([genre_vector_data, casts_vector_data, director_vector_data], axis=1)
	feature_vector = pd.concat([nums, vectors], axis=1).reset_index()
	feature_vector.drop('index', axis=1, inplace=True)
	vector = cosine_similarity(feature_vector)
	return vector


tmdb = TMDb()
tmdb.api_key = '7fbaefed20dc1346f16bfa3c1bcafb2b'
person = Person()
mov = Movie()

header = """<div style="padding:10px;text-align:centre"><h1 style="color:#c70039; font-size:45px">Movie Recommendation System</h1></div>"""
favicon = """<head>
  <link rel='icon' href='https://image.flaticon.com/icons/svg/1/1854.svg'>
</head>"""

html_tmp = """ 
<style>
body {
  background-image: url('https://demo.crea8social.com/storage/uploads/5576/2018/photos/profile/cover/_2000_c982226557be9f00d1b168af3da3b036.jpg');
  color: #f4f6ff;
}
</style> """

st.markdown(favicon, unsafe_allow_html=True)
st.markdown(html_tmp, unsafe_allow_html=True)
st.markdown(header, unsafe_allow_html=True)


movies = pd.read_csv('main_preprocessed.csv')
main = pd.read_csv('app.csv')

with st.spinner('Please wait...'):
	vector=extract_features(movies, main)

@st.cache(show_spinner=False)
def get_movie_details(main, name):

	name = name.lower()
	data = []
	if len(main[main.title == name]):
		data = main[main.title == name].sort_values('release_date').head(1)
	elif len(main[main.title.str.contains(name)]):
		data = main[main.title.str.contains(name)].sort_values('release_date').head(1)

	title = data.title.values[0]
	rating = float(data.imdb_rating.values[0])
	release_date = data.release_date.values[0]
	genres = data.genre.values[0]
	movie_id = mov.search(title)[0].id
	response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}&language=en-US').json()
	poster_img = None
	if 'poster_path' in response.keys() and response['poster_path']:
		img_bytes = requests.get('http://image.tmdb.org/t/p/w185'+response['poster_path']).content
		poster_img = Image.open(BytesIO(img_bytes)).resize((200, 250))
	else:
		img_bytes = requests.get('https://www.jakartaplayers.org/uploads/1/2/5/5/12551960/2297419_orig.jpg').content
		poster_img = Image.open(BytesIO(img_bytes)).resize((200, 250))

	return (title, release_date, rating, genres, poster_img)



@st.cache(show_spinner=False)
def get_recommendaion(name):

	movie_id = None
	if len(main[main.title == name]):
		movie_id = main[main.title == name].sort_values('release_date').head(1).movie_id.values[0]
	elif len(main[main.title.str.contains(name)]):
		movie_id = main[main.title.str.contains(name)].sort_values('release_date').head(1).movie_id.values[0]
	similar = pd.DataFrame(vector[movie_id], columns=['similarity'])
	similar_movies = similar.sort_values('similarity', ascending=False).index[1:11]
	similar_movies = main.loc[similar_movies].title.values
	similar_titles = [mov.title() for mov in list(similar_movies)]
	return set(similar_titles)


@st.cache(show_spinner=False)
def get_cast_details(cast):

	person_id = person.search(cast)[0].id
	response = requests.get(f'https://api.themoviedb.org/3/person/{person_id}?api_key={tmdb.api_key}&language=en-US').json()
	cast_img, cast_bio = None, None 
	if 'biography' in response.keys() and response['biography']:
		cast_bio = response['biography']
	else:
		cast_bio = 'Bio not available'
	if 'profile_path' in response.keys() and response['profile_path']:
		img_bytes = requests.get('http://image.tmdb.org/t/p/w185'+response['profile_path']).content
		cast_img = Image.open(BytesIO(img_bytes)).resize((150, 200))
	else:
		img_bytes = requests.get('https://www.diabetes.co.uk/forum/styles/uix/xenforo/avatars/avatar_female_l.png').content
		cast_img = Image.open(BytesIO(img_bytes)).resize((150, 200))
	return (cast_img, cast_bio)



search = st.text_input('', value='search')

if search != 'search':
	search = ' '.join(re.sub('[^a-zA-Z0-9]', ' ', search).lower().split())
	try:
		
		casts = []
		if len(main[main.title == search]):
			casts = main[main.title == search].sort_values('release_date').casts.values[0].split(',')
		elif len(main[main.title.str.contains(search)]):
			casts = main[main.title.str.contains(search)].sort_values('release_date').casts.values[0].split(',')

		cast_details = None
		cast_bios = {}

		with st.spinner('Please Wait....'):

			cast_images = []
			if casts[0] != 'unknown':
				cast_details = [get_cast_details(cast) for cast in casts]
				cast_images = [cast[0] for cast in cast_details]
				cast_bios = {cast: details[1] for cast, details in zip(casts, cast_details)}

			
			movie_title, movie_release, movie_rating, movie_genre, movie_img = get_movie_details(main, search)
			similar_titles = get_recommendaion(movie_title)
			similar_movie_details = {title: get_movie_details(main, title) for title in similar_titles}


			st.write('')
			st.write('')
			st.image(movie_img)
			st.markdown(f'Movie Name:  {movie_title.title()}')
			st.markdown(f'Release Date:  {movie_release}')
			st.markdown(f'IMDb Rating:  {movie_rating} :star:')
			st.markdown(f'Genre:  {movie_genre}')

			st.subheader('Top Casts')
			st.write('')
			st.image(cast_images, caption=casts)


		st.subheader('Bios')
		bio = st.selectbox('', ['None', casts[0], casts[1], casts[2]])
		bios_place = st.empty()
		see_more_place = st.empty()
		if bio != 'None':
			url = 'https://en.wikipedia.org/wiki/'+'_'.join(bio.split())
			bios_place.markdown(cast_bios[bio])
			see_more_place.markdown(f"<a href={url}>see more..</a>", unsafe_allow_html=True)

		st.subheader('Recommended Movies')
		st.write('')

		with st.spinner('Please Wait...'):
			for title, details in similar_movie_details.items():
				#movie_title, movie_release, movie_rating, movie_genre, movie_img
				movie_title = details[0]
				movie_release = details[1]
				movie_rating = details[2]
				movie_genre = details[3]
				movie_img = details[4]
				st.image(movie_img)
				st.markdown(f'Movie Name:  {movie_title.title()}')
				st.markdown(f'Release Date:  {movie_release}')
				st.markdown(f'IMDb Rating:  {movie_rating} :star:')
				st.markdown(f'Genre:  {movie_genre}')


	except:
		st.write('')
		st.write('')
		st.subheader('Sorry! Nothing Found..')

else:
	pass
