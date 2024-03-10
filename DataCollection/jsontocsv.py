import json
import pandas as pd
from progressbar import progressbar

# Read the JSON file into a Python object
with open('DataCollection/submissions.json', 'r') as file:
    data = json.load(file)

# Extracting data from the nested structure with a progress bar
list_of_dicts = [
    {
        'title': item['title'],
        'description': item['description'],
        'reference': item['reference'],
        'date': item['date'],
        'popularity': item['popularity'],
        # Concatenating comment number and content into a single string for each comment
        'comments': '\n'.join([f"comment {comment['number']}: {comment['content']}" for comment in item.get('comments', [])])
    }
    for item in progressbar(data) # Wrap the iteration with progressbar
]

# Creating a DataFrame from the extracted data
df = pd.DataFrame(list_of_dicts)

# Displaying the DataFrame
df.to_csv('DataCollection/csvdata.csv', index=False, encoding='utf-8')
