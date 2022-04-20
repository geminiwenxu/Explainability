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
    last_four_layers = token_embeddings[:, 0:4, :]
    # the_sum = torch.sum(last_four_layers, dim=0)
    # temp = the_sum.reshape(1, -1)
    # temp_1 = temp.squeeze()
    # vec_sum = temp_1.numpy()
    # the_min = torch.min(last_four_layers, dim=0)
    # temp = the_min[0].reshape(1, -1)
    # temp_1 = temp.squeeze()
    # vec_min = temp_1.numpy()
    the_mean = torch.mean(last_four_layers, dim=0)
    temp = the_mean.reshape(1, -1)
    temp_1 = temp.squeeze()
    vec_mean = temp_1.numpy()
    print(vec_mean.shape)

    # token_vecs_cat = []
    # token_vecs_sum = []
    # token_vecs_mean = []
    # token_vecs_max = []
    # token_vecs_min = []
    # for token in token_embeddings:
    #     cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
    #     token_vecs_cat.append(cat_vec)
    #     sum_vec = torch.sum(token[-4:], dim=0)
    #     token_vecs_sum.append(sum_vec)
    #     mean_vec = torch.mean(token[-4:], dim=0)
    #     token_vecs_mean.append(mean_vec)
    #     max_vec = torch.max(token[-4:], dim=0)
    #     token_vecs_max.append(max_vec)
    #     min_vec = torch.min(token[-4:], dim=0)
    #     token_vecs_min.append(min_vec)

    return vec_mean


if __name__ == "__main__":
    text = "hallo world"
    embeddings(text)
