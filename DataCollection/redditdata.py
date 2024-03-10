### Reddit Data 
import praw
import asyncpraw
import datetime
import json
from aiohttp import ClientSession

# Setup of reddit app 
reddit = praw.Reddit(client_id='PJxseFpHblIKsIPoDmWwyQ',
                     client_secret='En8dmoO9CLURm8mybarj9d2wDNl7EA',
                     user_agent='testing the apo end points')

print(reddit) # reddit object created 

#Output Data to Single Json file 

# Define the flairs you're interested in
flairs = ["Investieren - Aktien", "Investieren - ETF"]

# Get all submissions in the subreddit
submissions = []
for submission in reddit.subreddit('Finanzen').hot(limit=None):
    # Check if the post's flair is in the list of flairs
    if submission.link_flair_text in flairs:
        # Calculate the post's creation date
        created_utc = submission.created_utc
        post_created = datetime.datetime.fromtimestamp(created_utc)
        post_created = post_created.strftime("%Y%m%d")

        # Add the post and its creation date to the submissions list
        submissions.append((submission, post_created))

# Sort the submissions by their creation date in descending order
sorted_submissions = sorted(submissions, key=lambda s: s[1], reverse=True)

# Process each submission and add it to a list of submission dictionaries
submission_list = []
for i, (submission, post_created) in enumerate(sorted_submissions, start=1):
    # Your existing processing code here...
    # Print the title, selftext, and url
    title =  submission.title
    titletext = submission.selftext
    titleurl =  submission.url
    score = submission.score
    Popularity = score
    post = post_created

    # Replace "MoreComments" objects with actual Comment objects
    submission.comments.replace_more(limit=None)
    # Create a dictionary with the submission details
    submission_info = {
            'title': title,
            'description': titletext,
            'reference': titleurl,
            'date': post,
            'popularity': Popularity,
            'comments': [{'number': i+1, 'content': comment.body} for i, comment in enumerate(submission.comments.list())]
        }

    # Add the submission_info dictionary to the submission_list
    submission_list.append(submission_info)

# Write the submission_list to a single JSON file
with open("submissions.json", 'w') as json_file:
    json.dump(submission_list, json_file, indent=4)
    print("============jsoncreated==========")