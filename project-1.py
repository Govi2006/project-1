import pandas as pd

# Sample tweets
data = {
    "id": [1, 2, 3, 4, 5],
    "text": [
        "I'm so happy today! Everything is going great.",
        "I can't believe this is happening again. So frustrating.",
        "I miss my family a lot right now.",
        "This is the best day of my life!",
        "I'm worried about the results."
    ]
}

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("sample_tweets.csv", index=False)
print("sample_tweets.csv created successfully.")

import pandas as pd

df = pd.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "text": [
        "I'm so happy today! Everything is going great.",
        "I can't believe this is happening again. So frustrating.",
        "I miss my family a lot right now.",
        "This is the best day of my life!",
        "I'm worried about the results."
    ]
})
df.to_csv("sample_tweets.csv", index=False)
print("Created 'sample_tweets.csv'")
import pandas as pd
from transformers import pipeline

# --- Load Emotion Classifier ---
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=False)

# --- Load CSV and Analyze Emotions ---
def analyze_emotions_from_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Check for 'text' column
    if 'text' not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    results = []
    for index, row in df.iterrows():
        tweet = row['text']
        try:
            emotion = emotion_classifier(tweet)[0]
            results.append({
                "id": row.get("id", index),
                "Text": tweet,
                "Emotion": emotion["label"],
                "Confidence": round(emotion["score"], 3)
            })
        except Exception as e:
            print(f"Error analyzing tweet: {tweet}\n{e}")
            continue

    return pd.DataFrame(results)

# --- Main Program ---
if _name_ == "_main_":
    input_csv = "sample_tweets.csv"
    output_csv = "emotion_results.csv"

    print(f"Reading from: {input_csv}")
    result_df = analyze_emotions_from_csv(input_csv)

    print("\n--- Emotion Analysis ---")
    print(result_df.to_string(index=False))

    result_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")


