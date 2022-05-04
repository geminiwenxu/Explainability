import torch
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, BertModel


class Embedding:
    def __init__(self, vec_name, text, model_name='bert-base-uncased'):
        self.vec_name = vec_name
        self.text = text
        self.torch = torch
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.eval()

    def encode(self):
        marked_text = "[CLS] " + self.text + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = self.torch.tensor([indexed_tokens])
        segments_tensors = self.torch.tensor([segments_ids])
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]  # the third item will be the hidden states from all layers.
        token_embeddings = self.torch.stack(hidden_states, dim=0)
        token_embeddings = self.torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        last_four_layers = token_embeddings[:, -4:, :]
        print(last_four_layers.shape)

        if self.vec_name == 'max':
            vec = self.torch.max(last_four_layers, dim=0)
            temp = vec[0].reshape(1, -1)
        elif self.vec_name == 'min':
            vec = self.torch.min(last_four_layers, dim=0)
            temp = vec[0].reshape(1, -1)
        elif self.vec_name == 'mean':
            vec = self.torch.mean(last_four_layers, dim=0)
            temp = vec.reshape(1, -1)
        else:
            vec = self.torch.sum(last_four_layers, dim=0)
            temp = vec.reshape(1, -1)
        temp_1 = temp.squeeze()
        vec = temp_1.numpy()
        return vec.tolist()


class BertSentenceConverter:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.model.eval()

    def encode_to_vec(self, sentences, token=None, nlp=False):
        if type(sentences) == str:
            sentences = [sentences]

        for sentence in sentences:
            if len(sentence) > 0 and sentence[-1] != ".":
                sentence += "."

        embeddings = self.model.encode(sentences, convert_to_tensor=True)

        return embeddings.detach().cpu().numpy().tolist()


if __name__ == "__main__":
    text = "hallo world"
    test = Embedding('max', text)
    vec = test.encode()
    Embedding.encode(test)
    test_2 = BertSentenceConverter()
    xx = test_2.encode_to_vec(text)
    print(len(xx[0]))
