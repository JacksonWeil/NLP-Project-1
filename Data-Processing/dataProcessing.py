import pandas as pd
import spacy
from collections import Counter
import matplotlib.pyplot as plt
import requests

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
# Load spaCy model

nlp = spacy.load("en_core_web_sm")
novel="crystalstopper"
# Load the novel text
with open(novel+".txt", "r") as file:
    novel_text = file.read().lower()

# Apply spaCy processing
doc = nlp(novel_text)

# Extract named entities, focusing on characters
characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
character_freq = Counter(characters)
character_data = pd.DataFrame(character_freq.items(), columns=["Character", "Frequency"])

# Display the top characters for the report
print(character_data.sort_values(by="Frequency", ascending=False).head())

# Extract events (verbs) to capture action sequences
events = [token.lemma_ for token in doc if token.pos_ == "VERB"]
event_freq = Counter(events)
event_data = pd.DataFrame(event_freq.items(), columns=["Event", "Frequency"])

print(event_data.sort_values(by="Frequency", ascending=False).head())



# Plot character frequencies
plt.figure(figsize=(10, 5))
character_data.sort_values(by="Frequency", ascending=False).head(10).plot(kind="bar", x="Character", y="Frequency", legend=False)
plt.title("Top 10 Characters by Frequency")
plt.xlabel("Character")
plt.ylabel("Frequency")
plt.savefig(novel+"character_frequency.jpg", format="jpg")
plt.close()
# Plot event frequencies
plt.figure(figsize=(10, 5))
event_data.sort_values(by="Frequency", ascending=False).head(10).plot(kind="bar", x="Event", y="Frequency", legend=False)
plt.title("Top 10 Events by Frequency")
plt.xlabel("Event")
plt.ylabel("Frequency")
plt.savefig(novel+"events_frequency.jpg", format="jpg")
plt.close()



# Use CountVectorizer to transform text data
vectorizer = CountVectorizer(max_features=1000)
X = vectorizer.fit_transform([novel_text])
y = [1]  # A placeholder target for demonstration purposes

# Model example: Naive Bayes Classifier to analyze narrative structure
clf = MultinomialNB()
clf.fit(X, y)
predicted = clf.predict(X)

print(f"Predicted Plot Element Category: {predicted[0]}")


def generate_report(character_data, event_data):
    report = f"""
    Crime Novel Analysis Report

    Top Characters:
    {character_data.sort_values(by="Frequency", ascending=False).head(5).to_string(index=False)}

    Top Events:
    {event_data.sort_values(by="Frequency", ascending=False).head(5).to_string(index=False)}

    Basic Model Prediction (for demonstration purposes):
    Predicted plot element category: {predicted[0]}
    """
    with open(novel+"analysis_report.txt", "w") as file:
        file.write(report)


generate_report(character_data, event_data)

#

#
# def download_novel(url):
#     response = requests.get(url)
#     response.raise_for_status()  # Ensure the request was successful
#     return response.text
#
# # URL of "Arsène Lupin, Gentleman-Burglar"
# novel_url = 'http://www.gutenberg.org/cache/epub/6133/pg6133.txt'
# novel_text = download_novel(novel_url)
#
# # Save the downloaded text to a file
# with open("arsene_lupin.txt", "w") as file:
#     file.write(novel_text)