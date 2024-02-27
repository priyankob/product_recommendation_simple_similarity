import pandas as pd
import argparse
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def process_json_to_dataframe(file):
    with open(file, 'r', encoding="utf8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)

    return df

def recommend_items(items, item_id, num_recommendations=5):
    # Convert the list of items into a Pandas DataFrame
    df = items

    # Create a TF-IDF vectorizer to convert text attributes to numerical feature vectors
    vectorizer = TfidfVectorizer(stop_words="english")

    # Convert the 'name' and 'description' attributes to TF-IDF feature vectors
    status_vectors = vectorizer.fit_transform(df["status"])
    title_vectors = vectorizer.fit_transform(df["title"])
    content_vectors = vectorizer.fit_transform(df["content"])
    interest_vectors = vectorizer.fit_transform(df["interest"])
    item_name_vectors = vectorizer.fit_transform(df["item_name"])
    username_vectors = vectorizer.fit_transform(df["user.username"])
    item_url_vectors = vectorizer.fit_transform(df["item_url"])

    df["category"] = df["category"].fillna(0)
    df["item_unit_price"] = df["item_unit_price"].fillna(0)
    df["item_weight"] = df["item_weight"].fillna(0)

    # Calculate cosine similarities between items based on their attributes
    item_vectors = pd.concat(
        [
            pd.DataFrame(status_vectors.toarray()),
            pd.DataFrame(title_vectors.toarray()),
            pd.DataFrame(content_vectors.toarray()),
            pd.DataFrame(interest_vectors.toarray()),
            pd.DataFrame(item_name_vectors.toarray()),
            pd.DataFrame(username_vectors.toarray()),
            pd.DataFrame(item_url_vectors.toarray()),
            df[["category", "item_unit_price","item_weight"]],
        ],
        axis=1,
    )
    similarities = cosine_similarity(item_vectors)

    # Get the index of the item to recommend items similar to
    item_index = df.index[df["id"] == item_id][0]

    # Sort the similarities in descending order and get the top N items
    similar_items = list(enumerate(similarities[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[
        1 : num_recommendations
    ]
    similar_item_indices = []
    similar_item_indices.append(item_index)
    for si in similar_items:
        similar_item_indices.append(si[0])

    # Return the recommended items
    return similar_item_indices

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='Path to JSON file')
    parser.add_argument('--id', type=str, help='ID of search Item')
    args = parser.parse_args()

    if args.file:
        df = process_json_to_dataframe(args.file)
    else:
        print('Please specify a path to a JSON file using --file')
        
    if args.id:
        indices = recommend_items(df, args.id, num_recommendations=5)
        for i in indices:
            print(df.iloc[i]['id'],df.iloc[i]['title'] )
    else:
        print('Please specify a search item using --id')
        
        


