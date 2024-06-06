from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
import nltk
import csv
import pandas as pd
from sklearn.metrics import confusion_matrix
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')

# sia = SentimentIntensityAnalyzer()

# # Read the CSV file
# with open('genz_slang.csv', 'r', encoding='ISO-8859-1') as f:
#     reader = csv.reader(f)
#     next(reader)  # Skip the header row
#     new_words = {row[0]: float(row[2]) for row in reader if len(row) >= 3}

# sia.lexicon.update(new_words)

# data = pd.read_csv('training_set.tsv', delimiter='\t', encoding='iso-8859-1')
# sentences = data['sentence'].tolist()

# # Label the sentiment of the sentences
# sentiment_labels = []
# for sentence in sentences:
#     polarity_score = sia.polarity_scores(sentence)['compound']
#     if polarity_score > 0:
#         sentiment_labels.append('pos')
#     else:
#         sentiment_labels.append('neg')

# # Add the sentiment labels to the DataFrame
# data['sentiment'] = sentiment_labels

# data.to_csv('labeled_data.tsv', sep='\t', index=False)

#Training part
training_data = pd.read_csv('labeled_data.tsv', delimiter='\t', encoding='iso-8859-1')
train_sentences = training_data['sentence'].tolist()
train_labels = training_data['sentiment'].tolist()

# Transform the sentences into TF-IDF vectors
vectorizer = TfidfVectorizer()
train_vectors = vectorizer.fit_transform(train_sentences)

# Check the shape of the vectors
print(f'Train vectors shape: {train_vectors.shape}')

# Check the first vector
print(f'First train vector: {train_vectors[0].toarray()}')

# Encode labels
le = LabelEncoder()
train_labels_encoded = le.fit_transform(train_labels)

# Check the encoded labels
print(f'First 10 encoded labels: {train_labels_encoded[:10]}')

# Check the classes
print(f'Classes: {le.classes_}')

# Initialize an SVM classifier and train it on the labeled data
classifier = svm.SVC(kernel='linear')
classifier.fit(train_vectors, train_labels_encoded)

# Read the TSV file
df = pd.read_csv('test_set.txt', delimiter='\t', encoding='iso-8859-1')

# Transform the test sentences into TF-IDF vectors
test_vectors = vectorizer.transform(df['sentence'])

# Predict the labels for the test data
test_predictions = classifier.predict(test_vectors)

# Decode the predicted labels
test_predictions_decoded = le.inverse_transform(test_predictions)

test_predictions_encoded = le.transform(test_predictions_decoded)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(df['sentiment'], test_predictions_encoded, labels=[0, 1]).ravel()

# Calculate the accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f'Accuracy: {accuracy}')
print(f'True positives: {tp}')
print(f'False positives: {fp}')
print(f'True negatives: {tn}')
print(f'False negatives: {fn}')