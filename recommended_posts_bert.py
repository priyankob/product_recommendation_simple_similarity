import pandas as pd
import argparse
import json
import torch
from transformers import DistilBertModel, DistilBertTokenizer
from scipy.spatial.distance import cosine

def process_json_to_dataframe(file):
    with open(file, 'r', encoding="utf8") as f:
        data = json.load(f)
    df = pd.json_normalize(data)

    return df

def recommend_items(df, item_id, num_recommendations=5):
    # Define the list of items
    items = df["item_name"]
    # Define the input item
    input_item = df["item_name"][df.index[df['id']==item_id][0]]
    # Load the pre-trained DistilBERT model and tokenizer
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Convert the input item to a BERT input tensor
    input_tokens = tokenizer.encode(input_item, add_special_tokens=True)
    input_tensor = torch.tensor([input_tokens])

    # Get the BERT embeddings for the input item
    with torch.no_grad():
        input_embedding = model(input_tensor)[0][0, 0, :].numpy()


    # Calculate the cosine similarity between the input item embedding and the embeddings of all items in the list
    similarity_scores = []
    for item in items:
        item_tokens = tokenizer.encode(item, add_special_tokens=True)
        item_tensor = torch.tensor([item_tokens])
        with torch.no_grad():
            item_embedding = model(item_tensor)[0][0, 0, :].numpy()
        similarity_score = 1 - cosine(input_embedding, item_embedding)
        similarity_scores.append(similarity_score)

    similar_items = list(enumerate(similarity_scores))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)[0:5]
    similar_item_indices = []
    for si in similar_items:
        similar_item_indices.append(si[0])
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
        
        




