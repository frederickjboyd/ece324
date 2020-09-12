import argparse
from time import time
from torchsummary import summary

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import numpy as np

from model import SmallNet, BatchNorm, FourConvLayers, ReallySmallNet


# Set manual seed
seed = 1
torch.manual_seed(seed)

# Constants
SMALL_IMAGE_DIR = 'asl_images_personal'
FULL_IMAGE_DIR = 'asl_images'


def one_hot(x, dim):
    vec = torch.zeros(dim)
    vec[x] = 1.0
    return vec

# Convert images to tensors and normalize
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


# Define classes
classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K')


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# ENCODE CATEGORICAL FEATURES
# label_encoder = LabelEncoder()
# encoded_data = balanced_data[categorical_feats].apply(lambda col: label_encoder.fit_transform(col)).copy()
# income_col = encoded_data["income"].copy().to_numpy()
# del encoded_data["income"]
#
# oneh_encoder = OneHotEncoder(categories="auto")
# encoded_data = oneh_encoder.fit_transform(encoded_data).toarray()


# Load model
def load_model_small(lr):
    # model_small = SmallNet()
    # model_small = BatchNorm()
    # model_small = FourConvLayers()
    model_small = ReallySmallNet()
    loss_func = torch.nn.MSELoss()
    # loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_small.parameters(), lr)

    return model_small, loss_func, optimizer


def many_cold(one_hot, size):
    one_hot_list = one_hot.tolist()
    labels = []
    for i in range(len(one_hot_list)):
        labels.append(one_hot_list[i].index(max(one_hot_list[i])))
    return labels


def evaluate(model, val_loader):
    total_corr = 0

    for i, batch in enumerate(val_loader):
        features, label = batch
        # print("features:", features)
        # tensor_image = features[0]
        # plt.imshow(tensor_image.permute(1, 2, 0))
        # plt.title(label[0].item())
        # plt.show()
        # print("label:", label)

        # Run model on data
        prediction = model(features)
        # print("prediction:", prediction)

        # Check number of correct predictions
        # corr = (prediction > 0.5).squeeze().long() == label.long()
        torch_max = torch.max(prediction, 1)
        # print("torch.max:", torch_max)
        # print("labels:", label)
        for j in range(prediction.size()[0]):
            # print(torch_max[1][j].item())
            # print(label[j].item())
            # print(torch_max[1][j].item(), label[j].item())
            if torch_max[1][j].item() == label[j].item():
                total_corr += 1

        # Count number of correct predictions
        # total_corr += int(corr.sum())

    return float(total_corr) / len(val_loader.dataset)


def unencode(one_hot, size):
    one_hot_list = one_hot.tolist()
    labels = []
    for i in range(len(one_hot_list)):
        labels.append(one_hot_list[i].index(max(one_hot_list[i])))
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=10)

    args = parser.parse_args()

    t = 0

    dataset_small = torchvision.datasets.ImageFolder(FULL_IMAGE_DIR, transform=transform)
    train_data, valid_data = train_test_split(dataset_small, test_size=0.2, shuffle=True, random_state=seed)
    label_encoder = LabelEncoder()
    int_classes = label_encoder.fit_transform(classes)
    oneh_encoder = OneHotEncoder(categories="auto")
    int_classes = int_classes.reshape(-1, 1)
    oneh_labels = oneh_encoder.fit_transform(int_classes).toarray()
    # dataloader_small = torch.utils.data.DataLoader(dataset_small, batch_size=args.batch_size, shuffle=True)

    # Calculate mean and std
    # mean = 0
    # std = 0
    # for file, _ in dataloader_small:
    #     samples = file.size(0)
    #     file = file.view(samples, file.size(1), -1)
    #     mean += file.mean(2).sum(0)
    #     std += file.std(2).sum(0)
    # mean /= len(dataloader_small)
    # std /= len(dataloader_small)
    # print("MEAN:", mean)
    # print("STD:", std)

    dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)

    target_transform = transforms.Compose([transforms.Lambda(lambda x: one_hot(x, 10))])
    dataTest = torchvision.datasets.ImageFolder('asl_images', transform=transform, target_transform=target_transform)
    dummy_train_data, validDataConfuse = train_test_split(dataTest, test_size=0.2, random_state=seed)

    # dataloader_small = load_data_small(args.batch_size)
    model_small, loss_func, optimizer = load_model_small(args.lr)

    model_small = model_small.to("cpu")

    epochs = []
    gradient_steps_small = []
    training_accuracies_small = []
    validation_accuracies_small = []
    training_losses = []
    validation_losses = []
    times = []

    # Record time before entering training loop
    prev_time = time()

    for epoch in range(args.epochs):
        accum_loss = 0
        accum_loss_valid = 0
        num_batches = 0
        num_batches_valid = 0

        for i, batch in enumerate(dataloader_train):
            # Get batch of data
            features, label = batch

            num_batches += 1

            # Set gradients to zero
            optimizer.zero_grad()

            # Run neural network on batch
            predictions = model_small(features)
            # print("label:", label)

            # Compute loss
            batch_loss = loss_func(input=predictions.squeeze(), target=torch.Tensor(oneh_labels[label]))
            # batch_loss = loss_func(input=predictions.squeeze(), target=(label))

            accum_loss += batch_loss

            # Calculate gradients
            batch_loss.backward()
            optimizer.step()

            gradient_steps_small.append(t + 1)

            t += 1

        # Calculate validation losses
        for j, batch_valid in enumerate(dataloader_valid):
            num_batches_valid += 1

            # Get batch of data
            features_valid, label_valid = batch_valid

            # Run neural network on validation batch
            predictions_valid = model_small(features_valid)

            # Compute loss
            batch_loss_valid = loss_func(input=predictions_valid.squeeze(),
                                         target=torch.Tensor(oneh_labels[label_valid]))
            # batch_loss_valid = loss_func(input=predictions_valid.squeeze(),
            #                              target=(label_valid))

            accum_loss_valid += batch_loss_valid

        # Store epoch data in lists
        epochs.append(epoch)
        training_losses.append(accum_loss / num_batches)
        validation_losses.append(accum_loss_valid / num_batches_valid)
        training_acc = evaluate(model_small, dataloader_train)
        training_accuracies_small.append(training_acc)
        valid_acc = evaluate(model_small, dataloader_valid)
        validation_accuracies_small.append(valid_acc)
        times.append(time() - prev_time)
        print("epoch:", epoch, "training_acc:", training_acc, "valid_acc:", valid_acc)

    # Plot data
    plt.plot(epochs, training_accuracies_small, validation_accuracies_small)
    plt.title('Accuracies vs. Epoch')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracies', dpi=300)
    plt.show()

    plt.plot(epochs, training_losses, validation_losses)
    plt.title('Losses vs. Epoch')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('losses', dpi=300)
    plt.show()

    plt.plot(times, training_accuracies_small, validation_accuracies_small)
    plt.title('Training Time')
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Time [s]')
    plt.ylabel('Accuracy')
    plt.savefig('time', dpi=300)
    plt.show()

    print("Training Time:", time() - prev_time)
    print("Validation accuracy:", validation_accuracies_small[len(validation_accuracies_small) - 1])
    summary(model_small, (3, 56, 56), device="cpu")
    torch.save(model_small.state_dict(), 'MyBestSmall.pt')

    # Calculate confusion matrix
    inputs = [0 for i in range(len(validDataConfuse))]
    for i in range(len(inputs)):
        inputs[i] = validDataConfuse[i][0].tolist()

    inputs = torch.FloatTensor(inputs)
    outputs = model_small(inputs)

    labels = [validDataConfuse[i][1].tolist() for i in range(len(validDataConfuse))]
    labels = unencode(torch.FloatTensor(labels), 10)
    outputs = unencode(outputs, 10)

    print(confusion_matrix(labels, outputs))

    # Get random images
    # dataiter = iter(dataloader_small)
    # images, labels = dataiter.next()

    # Show images
    # imshow(torchvision.utils.make_grid(images))
    # Print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


if __name__ == '__main__':
    main()
