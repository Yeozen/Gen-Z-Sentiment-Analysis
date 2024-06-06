import nltk
from nltk.corpus import stopwords
import string
import csv
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from sklearn.metrics import confusion_matrix

nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

sia = SentimentIntensityAnalyzer()

# Read the CSV file
with open('genz_slang.csv', 'r', encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    new_words = {row[0]: float(row[2]) for row in reader if len(row) >= 3}

sia.lexicon.update(new_words)

# Define stop words and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Load the test set
df_test = pd.read_csv('test_set.txt', delimiter='\t', encoding='iso-8859-1')

# Calculate sentiment scores for the test set
df_test['Predicted_Sentiment_Score'] = df_test['sentence'].apply(lambda comment: sia.polarity_scores(comment)['compound'])

# Convert the sentiment scores into classes
df_test['Predicted_Sentiment_Class'] = df_test['Predicted_Sentiment_Score'].apply(lambda score: 1 if score > 0 else 0)
df_test['True_Sentiment_Class'] = df_test['sentiment'].apply(lambda score: 1 if score > 0 else 0)

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(df_test['True_Sentiment_Class'], df_test['Predicted_Sentiment_Class']).ravel()

# Calculate the accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f'Accuracy: {accuracy}')
print(f'True positives: {tp}')
print(f'False positives: {fp}')
print(f'True negatives: {tn}')
print(f'False negatives: {fn}')