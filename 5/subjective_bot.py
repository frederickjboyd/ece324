import torch

import torchtext
from torchtext import data

import spacy

BASE_DIR = 'models/'


def load_saved_models():
    baseline = torch.load(BASE_DIR + 'model_baseline.pt')
    cnn = torch.load(BASE_DIR + 'model_cnn.pt')
    rnn = torch.load(BASE_DIR + 'model_rnn.pt')

    for parameter in baseline.parameters():
        parameter.requires_grad = False

    for parameter in cnn.parameters():
        parameter.requires_grad = False

    for parameter in rnn.parameters():
        parameter.requires_grad = False

    baseline.eval()
    cnn.eval()
    rnn.eval()

    return baseline, cnn, rnn


def tokenizer(raw_sentence):
    spacy_en = spacy.load('en')
    return [tok.text for tok in spacy_en(raw_sentence)]


def categorize_pred(raw_prediction):
    if raw_prediction > 0.5:
        return 'subjective'
    return 'objective'


# 3.2.1
TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)

# # 3.2.2
train_data, val_data, test_data = data.TabularDataset.splits(
    path='data/', train='train.tsv',
    validation='validation.tsv', test='test.tsv', format='tsv',
    skip_header=True, fields=[('text', TEXT)])
#
# # 3.2.3
# train_iter, val_iter, test_iter = data.BucketIterator.splits(
#     (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
#     sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

# 3.2.4
TEXT.build_vocab(train_data, val_data, test_data)

# 4.1
TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
vocab = TEXT.vocab


baseline_model, cnn_model, rnn_model = load_saved_models()

sentence = input('Enter a sentence:')
tokens = tokenizer(sentence)
token_ints = [vocab.stoi[tok] for tok in tokens]
token_tensor = torch.LongTensor(token_ints).view(-1, 1)
lengths = torch.Tensor([len(token_ints)])

baseline_pred = baseline_model(token_tensor, lengths).item()
baseline_category = categorize_pred(baseline_pred)
cnn_pred = cnn_model(token_tensor, lengths).item()
cnn_category = categorize_pred(cnn_pred)
rnn_pred = rnn_model(token_tensor, lengths).item()
rnn_category = categorize_pred(rnn_pred)

print('Model baseline:', baseline_category, '(' + str(round(baseline_pred, 3)) + ')')
print('Model cnn:', cnn_category, '(' + str(round(cnn_pred, 3)) + ')')
print('Model rnn:', rnn_category, '(' + str(round(rnn_pred, 3)) + ')')
