import nltk
import praw
import csv
import unicodedata
from bs4 import BeautifulSoup
import re
from praw.models import MoreComments
import collections
import time
import prawcore

stopword_list=nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

# Your list of words
words = [
    'zesty', 'gyat'
]

# Initialize a dictionary to keep track of how many sentences you've obtained for each word
word_counts = collections.defaultdict(int)

# Initialize a counter for the total number of sentences
total_count = 0

# Define the quota for each word and the total quota
word_quota = 15
total_quota = 2000

# Compile a regex pattern for efficiency
pattern = re.compile('|'.join(r'\b%s\b' % w for w in words), re.IGNORECASE)

# Set up Reddit API credentials
reddit = praw.Reddit(client_id='UmAtpOuceYEwqziiClz_1A', client_secret='FOnfuDB1XqumpDHrcPU0hf1Amls9hg', user_agent='DIA PROJECT')

# Define the subreddits and sentiment analyzer
subreddits = ['gaming', 'GenZ', 'memes', 'AskReddit', 'youngadults', 'Instagramreality', 'TikTokCringe', 'GenZHumor', 'popculturechat', 'AskTeenBoys']
subreddits_more = ['gaming', 'GenZ', 'memes', 'AskReddit', 'youngadults', 'Instagramreality', 'TikTokCringe', 'GenZHumor', 'popculturechat', 'AskTeenBoys']

def remove_accent_chars(comment):
    comment = unicodedata.normalize('NFKD', comment).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return comment

def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

def comment_contains_word(comment):
    return pattern.search(comment) is not None

bot_disclaimer = "I am a bot, and this action was performed automatically. Please contact the moderators of this subreddit if you have any questions or concerns."


processed_comments = set()

with open('gyat.tsv', 'w', newline='', encoding='utf-8') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(['sentence'])  # write the header

    for subreddit in subreddits:
        for post in reddit.subreddit(subreddit).top(time_filter='all', limit=None):
            for _ in range(5):  # Retry up to 5 times
                try:
                    comments = post.comments
                except prawcore.exceptions.RequestException:
                    time.sleep(5)  # Wait for 5 seconds before retrying
                    continue
                break
            else:
                print("Failed to fetch comments after 5 attempts")
                continue
            # Assuming you have a list of comments
            for comment in comments:
                if isinstance(comment, MoreComments):
                    continue
                comment_text = remove_accent_chars(comment.body)
                comment_text = strip_html_tags(comment_text)
                if comment_contains_word(comment.body):
                    tsv_output.writerow([comment_text])
                    processed_comments.add(comment.id)

            for comment in comments:
                if comment.id in processed_comments:
                    continue
                processed_comments.add(comment.id)

                if isinstance(comment, MoreComments):
                    continue
                if comment.author is not None and comment.author.name.lower().endswith('bot'):
                    continue
                for word in words:
                    if word.lower() in comment.body.lower():
                        if word_counts[word] < word_quota:
                            # This comment contains the word and we haven't reached the quota for this word yet
                            # Now you can process it as before
                            processed_comment = remove_accent_chars(comment.body)
                            processed_comment = strip_html_tags(processed_comment)
                            tsv_output.writerow([processed_comment])  # write the comment to the file
                            print(processed_comment)
                            word_counts[word] += 1
                            total_count += 1
                            if total_count >= total_quota:
                                break
                if total_count >= total_quota:
                    break
            if total_count >= total_quota:
                break