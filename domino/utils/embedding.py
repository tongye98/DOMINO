from transformers import AutoModel, AutoTokenizer
import json 
from tqdm import tqdm 
import numpy as np
import torch 


def extract_embeddings(checkpoint, data_path, text_key="question_content", output_path=None):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    dataset = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            dataset.append(item[text_key])

    print(f"Dataset size: {len(dataset)}")

    all_embeddings = []
    for item in tqdm(dataset):
        inputs = tokenizer.encode(item, max_length=4096, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = model(inputs)[0]
        all_embeddings.append(embedding.cpu().numpy())

    all_embeddings_np = np.vstack(all_embeddings)
    print(f"Embedding shape: {all_embeddings_np.shape}")

    if output_path:
        np.save(output_path, all_embeddings_np)
        print(f"Saved to {output_path}")

    return all_embeddings_np


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="Salesforce/codet5p-110m-embedding")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--text_key", type=str, default="synthetic_text")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    extract_embeddings(
        checkpoint=args.checkpoint,
        data_path=args.data_path,
        text_key=args.text_key,
        output_path=args.output_path,
    )
