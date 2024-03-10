import csv
import sys
import time
from progressbar import ProgressBar
import os

# Define the directories 
csv_data = 'DataCollection/csvdata.csv'
directory = 'IngestionData'

def process_comments(comments):
    # Strip leading and trailing whitespace from each comment
    # and join them into a single string with a space character between each comment
    return " ".join(comment.strip() for comment in comments)

# Ensure the directory exists, create it if it doesn't
if not os.path.exists(directory):
    os.makedirs(directory)

# Initialize the progress bar
pbar = ProgressBar(maxval=100)
pbar.start()

with open(csv_data, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    total_rows = sum(1 for row in reader) # Count total rows to calculate progress
    csvfile.seek(0) # Reset file pointer to start
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        # Update progress bar
        progress = (i / total_rows) * 100
        pbar.update(progress)

        # Construct the file name using date and popularity
        file_name = f"date{row['date']}_popularity{row['popularity']}.md"

        # Specify the full path to the file within the IngestionData directory
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'w') as mdfile:
            # Title section
            mdfile.write(f"# {row['title']}\n\n")

            # Description section
            mdfile.write(f"## Description\n\n{row['description']}\n\n")

            # Reference (URL) section
            mdfile.write(f"## Reference\n\n[Reference]({row['reference']})\n\n")

            # Popularity section
            mdfile.write(f"## Popularity\n\n{row['popularity']}\n\n")

            # Date section
            mdfile.write(f"## Date\n\n{row['date']}\n\n")

            # Comments section
            comments = row['comments'].split(',') # Assuming comments are comma-separated
            comment_block = process_comments(comments)
            # Make the comments bold under a different header
            mdfile.write(f"### Comments\n\n**{comment_block}**\n\n")

pbar.finish()
