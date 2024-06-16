import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wordcloud import WordCloud
from bs4 import BeautifulSoup
import requests

from imdb import IMDb
import inquirer


def search_movie():
    ia = IMDb()
    movie_title = input("영화 제목을 입력하세요: ")

    # 영화 검색
    search_results = ia.search_movie(movie_title)

    if not search_results:
        print("검색 결과가 없습니다.")
        return

    # 검색 결과 출력
    choices = []
    for i, movie in enumerate(search_results):
        title = movie.get('title')
        year = movie.get('year')
        choices.append((f"{title} ({year}) - movieID: {movie.movieID}", movie.movieID))

    questions = [
        inquirer.List('movie',
                      message="영화를 선택하세요",
                      choices=choices,
                      ),
    ]
    answers = inquirer.prompt(questions)

    selected_movie_id = answers['movie']
    selected_movie = next(movie for movie in search_results if movie.movieID == selected_movie_id)

    print(f"선택한 영화: {selected_movie.get('title')} ({selected_movie.get('year')}) - movieID: {selected_movie.movieID}")
    return selected_movie.movieID


def scrape_imdb_reviews(movie_id, num_reviews=50):
    reviews = []
    sentiments = []
    url = f'https://www.imdb.com/title/{movie_id}/reviews?ref_=tt_ql_3'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    review_divs = soup.find_all('div', class_='text show-more__control')
    for review_div in review_divs[:num_reviews]:
        reviews.append(review_div.text)
        if len(review_div.text) > 200:
            sentiments.append('positive')
        else:
            sentiments.append('negative')

    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})


imdb_movie_id = search_movie()
data = scrape_imdb_reviews("tt" + imdb_movie_id, 100)

data.dropna(inplace=True)
data['review'] = data['review'].str.replace('<br />', ' ')
data['review'] = data['review'].str.replace('[^a-zA-Z]', ' ')
data['review'] = data['review'].str.lower()

vectorizer = CountVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(data['review'])
y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("정확성: ", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

data['sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

data['review_length'] = data['review'].apply(len)
data['review_length'].plot(kind='hist', bins=50)
plt.title('Review Length Distribution')
plt.xlabel('Length of Review')
plt.ylabel('Count')
plt.show()

positive_reviews = ' '.join(data[data['sentiment'] == 'positive']['review'])
negative_reviews = ' '.join(data[data['sentiment'] == 'negative']['review'])

wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(negative_reviews)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.title('Positive Reviews Word Cloud')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.title('Negative Reviews Word Cloud')
plt.axis('off')
plt.show()
