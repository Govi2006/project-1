import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data (replace with real data from Twitter, Reddit, etc.)
data = {
    'id': [1, 2, 3, 4, 5],
    'text': [
        "I am so happy with my results!",
        "This makes me so angry!",
        "I'm feeling really sad and down today.",
        "I'm excited about the upcoming concert!",
        "I'm scared about the exam tomorrow."
    ]
}

# Load data into DataFrame
df = pd.DataFrame(data)

# Define emotion lexicon (expand this for better accuracy)
emotion_keywords = {
    'happy': ['happy', 'joy', 'glad', 'pleased', 'excited'],
    'sad': ['sad', 'down', 'unhappy', 'depressed'],
    'angry': ['angry', 'mad', 'furious'],
    'fear': ['scared', 'afraid', 'fear', 'terrified'],
    'surprise': ['surprised', 'amazed', 'shocked']
}

# Function to classify emotion based on keyword match
def classify_emotion(text):
    text = text.lower()
    for emotion, keywords in emotion_keywords.items():
        if any(word in text for word in keywords):
            return emotion
    return 'neutral'

# Apply emotion classifier
df['emotion'] = df['text'].apply(classify_emotion)

# Display results
print(df)

# Plot emotion distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='emotion', palette='Set2')
plt.title('Emotion Distribution')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
