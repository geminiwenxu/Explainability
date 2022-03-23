import torch
from transformers import BertTokenizer, BertModel


def embeddings(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # marked_text = "[CLS] " + text + " [SEP]"
    marked_text = text
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens, len(indexed_tokens))

    # Mark each of the 22 tokens as belonging to sentence "1".
    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    # print(tokens_tensor)
    segments_tensors = torch.tensor([segments_ids])
    # print(segments_tensors)
    model = BertModel.from_pretrained('bert-base-uncased',
                                      output_hidden_states=True,  # Whether the model returns all hidden-states.
                                      )
    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Run the text through BERT, and collect all of the hidden states produced from all 12 layers.
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]  # the third item will be the hidden states from all layers.
    # print("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
    layer_i = 0
    # print("Number of batches:", len(hidden_states[layer_i]))
    batch_i = 0
    # print("Number of tokens:", len(hidden_states[layer_i][batch_i]))
    token_i = 0
    # print("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

    # Grouping the values by token:
    # Concatenate the tensors for all layers. We use `stack` here to create a new dimension in the tensor.
    token_embeddings = torch.stack(hidden_states, dim=0)
    # Remove dimension 1, the "batches".
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    # Swap dimensions 0 and 1. switch the layers and tokens dimensions.
    token_embeddings = token_embeddings.permute(1, 0, 2)
    # print(token_embeddings.size())  # torch.Size([22, 13, 768]) 22个tokens, 13层layers and 786个hidden states

    # token_vecs_cat = []
    # for token in token_embeddings:
    #     # concatenate the last four layers, giving us a single word vector per token.
    #     # Each vector will have length 4 x 768 = 3,072
    #     cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    #     token_vecs_cat.append(cat_vec)
    # # print(token_vecs_cat)
    # print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))  # Shape is: 22 x 3072
    # return token_vecs_cat, tokenized_text

    ######################################################
    # Stores the token vectors, with shape [22 x 768]
    token_vecs_sum = []

    # `token_embeddings` is a [22 x 12 x 768] tensor.

    # For each token in the sentence...
    for token in token_embeddings:
        print(token.shape)
        # `token` is a [12 x 768] tensor

        # Sum the vectors from the last four layers.
        sum_vec = torch.mean(token[-4:], dim=0)

        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec)
    print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))
    # print(token_vecs_sum)
    return token_vecs_sum, tokenized_text


if __name__ == "__main__":
    text = "hallo world"
    embeddings(text)
