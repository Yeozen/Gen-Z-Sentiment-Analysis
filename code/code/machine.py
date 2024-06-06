import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

def sentiment_labels(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return config.id2label[ranking[0]]

# Read the TSV file
df = pd.read_csv('test_set.txt', delimiter='\t', encoding='iso-8859-1')

# Create lists to store the predictions and actual labels
predictions = []
actual_labels = []

for index, row in df.iterrows():
    text = row['sentence']
    sentiment = sentiment_labels(text)

    # Include neutral results as false negatives or false positives
    if sentiment == 'neutral':
        if row['sentiment'] == 1:
            sentiment = 'negative'  # Count as false negative
        elif row['sentiment'] == 0:
            sentiment = 'positive'  # Count as false positive

    predictions.append(sentiment)
    actual_labels.append(row['sentiment'])

predictions_num = [0 if label == 'negative' else 1 for label in predictions]

# Calculate the confusion matrix
tn, fp, fn, tp = confusion_matrix(actual_labels, predictions_num, labels=[0, 1]).ravel()

# Calculate the accuracy
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f'Accuracy: {accuracy}')
print(f'True positives: {tp}')
print(f'False positives: {fp}')
print(f'True negatives: {tn}')
print(f'False negatives: {fn}')