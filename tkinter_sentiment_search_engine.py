import praw
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, scrolledtext
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, pipeline
from collections import Counter
from datetime import datetime

# Path to my local directory containing the model
model_path = "C:/Users/micha/Desktop/model/db_model" # Change this to your directory if replicating this code

# Load the model and tokenizer
try:
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    print("Model and tokenizer loaded successfully!") # Check message I used to ensure everything was working
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Initialize the sentiment analysis pipeline
try:
    sentiment_analyzer = pipeline("text-classification", model=model, tokenizer=tokenizer)
    print("Sentiment analysis pipeline created successfully!")
except Exception as e:
    print(f"Error initializing pipeline: {e}")

# Reddit credentials
import reddit_credentials as rc

reddit = praw.Reddit(
    client_id=rc.client_id,
    client_secret=rc.client_secret,
    user_agent=rc.user_agent,
    username=rc.username,
    password=rc.password
)

# Subreddit mapping for 32 NFL teams
team_to_subreddit = {
     "eagles": "eagles",
    "patriots": "Patriots",
    "cowboys": "cowboys",
    "buccaneers":"buccaneers",
    "bears":"CHIBears",
    "packers":"GreenBayPackers",
    "cardinals": "cardinals",
    "ravens": "ravens",
    "falcons": "falcons",
    "saints": "Saints",
    "bengals": "bengals",
    "bills": "buffalobills",
    "panthers": "panthers",
    "dolphins": "miamidolphins",
    "vikings": "minnesotavikings",
    "browns": "Browns",
    "broncos": "DenverBroncos",
    "lions": "detroitlions",
    "texans": "Texans",
    "colts": "Colts",
    "giants": "NYGiants",
    "jaguars": "Jaguars",
    "chiefs": "KansasCityChiefs",
    "patriots": "Patriots",
    "jets": "nyjets",
    "raiders": "raiders",
    "steelers": "steelers",
    "chargers": "Chargers",
    "49ers": "49ers",
    "seahawks": "Seahawks",
    "rams": "LosAngelesRams",
    "titans": "Tennesseetitans",
    "commanders": "Commanders"
}

def generate_keywords(player_name):
    return [player_name] + player_name.split()

def get_comments_with_keywords(reddit, subreddit_name, keywords, total_comments=1000):
    subreddit = reddit.subreddit(subreddit_name)
    filtered_comments = []
    for comment in subreddit.comments(limit=total_comments):
        try:
            if any(keyword.lower() in comment.body.lower() for keyword in keywords):
                filtered_comments.append({
                    'Comment': comment.body,
                    'Created_UTC': comment.created_utc,
                    'Comment_ID': comment.id
                })
        except Exception as e:
            continue
    return filtered_comments

def get_comments_from_subreddits(reddit, subreddits, keywords, total_comments_per_subreddit=1000):
    all_comments = []
    for subreddit_name in subreddits:
        subreddit_comments = get_comments_with_keywords(reddit, subreddit_name, keywords, total_comments_per_subreddit)
        all_comments.extend(subreddit_comments)
    return all_comments

def analyze_sentiment(comments):
    sentiment_results = sentiment_analyzer([comment['Comment'] for comment in comments])
    label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
    results = []
    timestamps = []

    for comment, result in zip(comments, sentiment_results):
        sentiment = label_map[result['label']]
        results.append({
            'Comment': comment['Comment'],
            'Sentiment': sentiment,
            'Score': result['score']
        })
        timestamps.append(comment['Created_UTC'])
    
    sentiment_counts = Counter([res['Sentiment'] for res in results])
    avg_timestamp = sum(timestamps) / len(timestamps) if timestamps else 0
    avg_datetime = datetime.utcfromtimestamp(avg_timestamp).strftime('%Y-%m-%d %H:%M:%S') if avg_timestamp else "N/A"
    
    most_common_sentiment = sentiment_counts.most_common(1)[0] if sentiment_counts else ("None", 0)

    return sentiment_counts, results, most_common_sentiment

# GUI Logic
def perform_analysis():
    player_name = player_name_entry.get().strip()
    team_name = team_name_entry.get().strip().lower()
    
    if not player_name:
        result_display.insert(tk.END, "Please enter a player's name.\n")
        return

    keywords = generate_keywords(player_name)
    subreddits = ["nfl", "fantasyfootball", "nflmemes", "espn"]
    
    if team_name in team_to_subreddit:
        subreddits.append(team_to_subreddit[team_name])

    result_display.insert(tk.END, f"Fetching comments for {player_name}...\n")
    comments = get_comments_from_subreddits(reddit, subreddits, keywords)

    if not comments:
        result_display.insert(tk.END, f"No comments found for {player_name}.\n")
        return

    sentiment_counts, results, most_common_sentiment = analyze_sentiment(comments)
    
    # Display our sentiment results
    result_display.insert(tk.END, "\nSentiment Analysis Results:\n")
    for sentiment, count in sentiment_counts.items():
        result_display.insert(tk.END, f"{sentiment}: {count}\n")
    
    result_display.insert(tk.END, f"\nMost Common Sentiment: {most_common_sentiment[0]} ({most_common_sentiment[1]} occurrences)\n")
    
    # Generate a bar graph
    if sentiment_counts:
        plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color=['blue', 'green', 'red'])
        plt.title(f'Sentiment Analysis for {player_name}')
        plt.xlabel("Sentiment")
        plt.ylabel("Count")
        plt.show()

def clear_inputs():
    player_name_entry.delete(0, tk.END)
    team_name_entry.delete(0, tk.END)
    result_display.delete(1.0, tk.END)

# Tkinter GUI
root = tk.Tk()
root.title("Sentiment Search Engine - Fantasy Football")

ttk.Label(root, text="Player Name:").grid(row=0, column=0, padx=10, pady=5, sticky=tk.W)
player_name_entry = ttk.Entry(root, width=30)
player_name_entry.grid(row=0, column=1, padx=10, pady=5)

ttk.Label(root, text="Team Name:").grid(row=1, column=0, padx=10, pady=5, sticky=tk.W)
team_name_entry = ttk.Entry(root, width=30)
team_name_entry.grid(row=1, column=1, padx=10, pady=5)

analyze_button = ttk.Button(root, text="Sentiment Search", command=perform_analysis)
analyze_button.grid(row=2, column=0, columnspan=2, pady=10)

clear_button = ttk.Button(root, text="Clear Inputs", command=clear_inputs)
clear_button.grid(row=3, column=0, columnspan=2, pady=10)

result_display = scrolledtext.ScrolledText(root, width=60, height=20, wrap=tk.WORD)
result_display.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

root.mainloop()