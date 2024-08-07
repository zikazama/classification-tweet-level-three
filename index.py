# Classification Tweet Level 3
# Author : Fauzi Fadhlurrohman

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Step 1 Train Classification

## Assume labeled_data is a DataFrame with 'tweet' and 'level' columns
X = labeled_data['tweet']
y = labeled_data['level']

vectorizer = TfidfVectorizer(max_features=1000)
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

## Predict on new data
y_pred = classifier.predict(X_test)

# Step 2 Evaluation

print(classification_report(y_test, y_pred))

# Step 3 Implementation

new_tweets = ["Contoh tweet baru yang perlu diklasifikasikan"]
new_tweets_clean = [clean_tweet(tweet) for tweet in new_tweets]
new_tweets_vectorized = vectorizer.transform(new_tweets_clean)

new_predictions = classifier.predict(new_tweets_vectorized)
