import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import numpy as np
from skimage.io import imread

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

from image_loaders import GleasonDataset
from resnet import resnet50, resnet101, resnet34

parser = argparse.ArgumentParser()

parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--which_resnet", type=str, default="resnet34")
parser.add_argument("--use_fcn", type=bool, default=False)
parser.add_argument("--patch_size", type=int, default=299)
parser.add_argument("--model_path", type=str, default="/home/fi5666wi/Documents/Python/torchProject/nldl/saved_models/"
                                                      "simclr_2023-02-16_ep40_model.pth")
parser.add_argument("--n_clusters", type=int, default=10)

args = parser.parse_args()


def get_model():
    if args.which_resnet == 'resnet34':
        model = resnet34(args)
    elif args.which_resnet == 'resnet50':
        model = resnet50(args)
    elif args.which_resnet == 'resnet101':
        model = resnet101(args)
    else:
        NotImplementedError("Architecture not implemented")

    model_weights = torch.load(args.model_path)
    model.load_state_dict(model_weights, strict=True)
    return model


def train_cluster(model, train_data, labels, patch_size=299, batch_size=10):
    dataset = GleasonDataset(train_data, labels=labels, mode='pca', patch_size=patch_size)
    tr_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)

    N = 512
    all_embeddings = np.zeros((len(dataset), N))
    all_targets = np.zeros(len(dataset))

    for i, (input, targets) in enumerate(tr_loader):
        with torch.no_grad():
            X = model(input)

        X_embedded = X  # TSNE(n_components=N, learning_rate='auto', method='exact',
        #   init = 'random', perplexity = 3).fit_transform(X)

        batch_sz = X_embedded.shape[0]

        all_embeddings[i * batch_sz:(i + 1) * batch_sz, :] = X_embedded
        all_targets[i * batch_sz:(i + 1) * batch_sz] = targets
        print('processed {} out of {}'.format(i+1, len(tr_loader)))

    centroids = np.zeros((len(labels), N))
    for y, label in enumerate(labels):
        centroids[y, :] = centroid(all_embeddings[all_targets==y])
    return centroids


def centroid(embedding):
    n_samples, n_dims = embedding.shape
    cent = np.zeros((1, n_dims))
    for dim in range(n_dims):
        sum = np.sum(embedding[:, dim])
        cent[0, dim] = sum/n_samples
    return cent


def binary_classify(model, data, clusters, labels, patch_size=299, threshold=0.5):
    dataset = GleasonDataset(data, labels=labels, mode='pca', patch_size=patch_size)
    data_loader = DataLoader(dataset, batch_size=1, drop_last=False)

    scores = np.zeros(len(data_loader))
    y_pred = np.zeros(len(data_loader))
    y_true = np.zeros(len(data_loader))
    for j, (image, y) in enumerate(data_loader):
        with torch.no_grad():
            x = model(image)

        d_b = np.linalg.norm(x-clusters[labels.index('benign'), :])
        d_m = np.linalg.norm(x-clusters[labels.index('malignant'), :])
        score = d_b/(d_m + d_b)

        scores[j] = score
        y_true[j] = y
        if score > threshold:
            y_pred[j] = 1

    return scores, y_true, y_pred


if __name__ == "__main__":
    sz = args.patch_size
    trainsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/test_' + str(sz)]

    testset = ['/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/test_' + str(sz)]

    labels = ['benign', 'malignant']
    colors = ['green', 'red', 'orange', 'purple']

    model = get_model()
    clusters = train_cluster(model, train_data=trainsets, labels=labels)
    _, targets, predictions = binary_classify(model, testset, clusters, labels)
    total_acc = accuracy_score(targets, predictions)

    print('Total Accuracy: {}'.format(total_acc))

    #plt.show()







