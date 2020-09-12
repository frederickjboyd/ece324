import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os

import matplotlib.pyplot as plt

from models import *


def load_model_baseline(lr, embedding_dim, vocab):
    model = Baseline(embedding_dim, vocab)
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_func, optimizer


def load_model_cnn(lr, embedding_dim, vocab, n_filters, filter_sizes):
    model = CNN(embedding_dim, vocab, n_filters, filter_sizes)
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_func, optimizer


def load_model_rnn(lr, embedding_dim, vocab, hidden_dim):
    model = RNN(embedding_dim, vocab, hidden_dim)
    loss_func = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_func, optimizer


def evaluate(model, generic_iter, loss_func):
    total_corr = 0
    batch_num = 0
    accum_loss = 0

    for i, batch in enumerate(generic_iter):
        batch_input, batch_input_length = batch.text
        batch_labels = batch.label
        batch_num += 1

        # Run model on data
        prediction = model(batch_input, batch_input_length)
        batch_loss = loss_func(input=prediction.squeeze(), target=batch_labels.float())
        accum_loss += batch_loss

        # Check number of correct predictions
        corr = (prediction > 0.5).squeeze().long() == batch_labels.long()

        # Count number of correct predictions
        total_corr += int(corr.sum())

    # Calculate average loss
    average_loss = accum_loss / batch_num

    return float(total_corr) / len(generic_iter.dataset), average_loss


def main(args):
    # Set seed
    torch.manual_seed(args.seed)

    ######
    # 3.2 Processing of the data
    # the code below assumes you have processed and split the data into
    # the three files, train.tsv, validation.tsv and test.tsv
    # and those files reside in the folder named "data".
    ######

    # 3.2.1
    TEXT = data.Field(sequential=True, lower=True, tokenize='spacy', include_lengths=True)
    LABELS = data.Field(sequential=False, use_vocab=False)

    # 3.2.2
    train_data, val_data, test_data = data.TabularDataset.splits(
            path='data/', train='train.tsv',
            validation='validation.tsv', test='test.tsv', format='tsv',
            skip_header=True, fields=[('text', TEXT), ('label', LABELS)])

    # 3.2.3
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
        sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)
    # train_iter, val_iter, test_iter = data.Iterator.splits(
    #     (train_data, val_data, test_data), batch_sizes=(args.batch_size, args.batch_size, args.batch_size),
    #     sort_key=lambda x: len(x.text), device=None, sort_within_batch=True, repeat=False)

    # Overfit data and iterator
    # overfit_data = data.TabularDataset(path='data/overfit.tsv', format='tsv', skip_header=True,
    #                                    fields=[('text', TEXT), ('label', LABELS)])
    # overfit_iter = data.BucketIterator(overfit_data, args.batch_size, sort_key=lambda x: len(x.text), device=None,
    #                                    sort_within_batch=True, repeat=False)

    # 3.2.4
    TEXT.build_vocab(train_data, val_data, test_data)

    # 4.1
    TEXT.vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))
    vocab = TEXT.vocab

    print("Shape of Vocab:", TEXT.vocab.vectors.shape)

    # 4.3
    dim = 100
    # model_baseline, loss_func, optimizer = load_model_baseline(args.lr, args.emb_dim, vocab)
    # model_baseline, loss_func, optimizer = load_model_cnn(
    #     args.lr, args.emb_dim, vocab, args.num_filt, args.rnn_hidden_dim
    # )
    model_baseline, loss_func, optimizer = load_model_rnn(args.lr, args.emb_dim, vocab, args.rnn_hidden_dim)

    t = 0
    gradient_steps = []
    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    training_losses = []
    validation_losses = []
    test_losses = []
    overfit_accuracies = []
    overfit_losses = []

    for epoch in range(args.epochs):
        accum_loss = 0
        num_correct = 0
        batch_num = 0

        for i, batch in enumerate(train_iter):
            batch_num += 1

            # Get one batch of data
            batch_input, batch_input_length = batch.text
            batch_labels = batch.label

            # Set gradients to zero
            optimizer.zero_grad()

            # Get predictions
            predictions = model_baseline(batch_input, batch_input_length)

            # Compute loss
            batch_loss = loss_func(input=predictions.squeeze(), target=batch_labels.float())
            accum_loss += batch_loss

            # Calculate gradients
            batch_loss.backward()

            # Update parameters
            optimizer.step()

            # Count number of correct predictions
            corr = (predictions > 0.5).squeeze().long() == batch_labels.long()
            num_correct += int(corr.sum())

        # Evaluate model every eval_every steps
        if (t + 1) % args.eval_every == 0:
            gradient_steps.append(t + 1)

            train_acc, train_loss = evaluate(model_baseline, train_iter, loss_func)
            # overfit_acc, overfit_loss = evaluate(model_baseline, overfit_iter, loss_func)
            valid_acc, valid_loss = evaluate(model_baseline, val_iter, loss_func)
            test_acc, test_loss = evaluate(model_baseline, train_iter, loss_func)
            print("Epoch: {}, Step {} | Loss: {} | Valid acc: {}".format(epoch+1, t+1, accum_loss / args.eval_every,
                                                                         valid_acc))
            training_accuracies.append(train_acc)
            training_losses.append(train_loss)
            # overfit_accuracies.append(overfit_acc)
            # overfit_losses.append(overfit_loss)
            validation_accuracies.append(valid_acc)
            validation_losses.append(valid_loss)
            test_accuracies.append(test_acc)
            test_losses.append(test_loss)

        t += 1

    final_index = len(training_accuracies) - 1
    print('Final Training Accuracy: ', round(training_accuracies[final_index], 3))
    print('Final Training Loss: ', round(training_losses[final_index].item(), 3))
    print('Final Validation Accuracy: ', round(validation_accuracies[final_index], 3))
    print('Final Validation Loss: ', round(validation_losses[final_index].item(), 3))
    print('Final Test Accuracy: ', round(test_accuracies[final_index], 3))
    print('Final Test Loss: ', round(test_losses[final_index].item(), 3))

    plt.plot(gradient_steps, training_accuracies)
    plt.plot(gradient_steps, validation_accuracies)
    # plt.plot(gradient_steps, overfit_accuracies)
    plt.legend(['Training', 'Validation'])
    # plt.legend(['Training'])
    plt.title('Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    plt.plot(gradient_steps, training_losses)
    plt.plot(gradient_steps, validation_losses)
    # plt.plot(gradient_steps, overfit_losses)
    plt.legend(['Training', 'Validation'])
    # plt.legend(['Training'])
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # torch.save(model_baseline, 'model_rnn.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    main(args)
