from transformers import BertModel, BertTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Example text
text = "Here is an example sentence."

# Tokenize the text
# This also adds the special tokens that BERT requires
input_ids = tokenizer.encode(text, add_special_tokens=True)

# Convert the list of IDs to a tensor
input_ids = torch.tensor([input_ids])

# Disable gradient calculation to save memory and speed up
with torch.no_grad():
    outputs = model(input_ids)

    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # The last hidden state is the embeddings for each token in the input
    last_hidden_states = outputs.last_hidden_state

    # To get a single vector for the entire input text, you can average the token vectors
    # You might want to exclude [CLS] and [SEP] tokens for this aggregation
    embeddings = last_hidden_states.mean(dim=1)

print("cls_embedding: ", cls_embedding)
print("embeddings: ", embeddings)