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


class SimpleDataset(Dataset):
    def __init__(self,
                 paths,
                 label):

        self.image_paths = self._load_img_paths(paths, label)
        self.input_dtype = torch.float32

    def _load_img_paths(self, paths, label):
        input_img_paths = []

        for dataset in paths:
            img_dir = os.path.join(dataset, label)
            this_input_paths = sorted([
                os.path.join(img_dir, fname)
                for fname in os.listdir(img_dir)
                if fname.endswith(".jpg")
            ])
            input_img_paths = input_img_paths + this_input_paths

        return input_img_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Generate one sample of data
        image = imread(self.image_paths[index])
        image = np.moveaxis(image.squeeze(), source=-1, destination=0)
        image = torch.from_numpy(image)

        return image.type(self.input_dtype)

parser = argparse.ArgumentParser()

parser.add_argument("--num_classes", type=int, default=2)
parser.add_argument("--which_resnet", type=str, default="resnet34")
parser.add_argument("--use_fcn", type=bool, default=False)
parser.add_argument("--patch_size", type=int, default=299)
parser.add_argument("--model_path", type=str, default="/home/fi5666wi/Documents/Python/torchProject/nldl/saved_models/"
                                                      "simclr_2023-02-20_ep70_model.pth")
parser.add_argument("--n_clusters", type=int, default=10)

args = parser.parse_args()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


if __name__ == "__main__":
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

    sz = args.patch_size
    trainsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Helsingborg/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Linkoping/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/train_' + str(sz)]

    testset = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/test_' + str(sz)]

    labels = ['benign', 'malignant']
    #labels = ['benign', 'G3', 'G4', 'G5']
    colors = ['green', 'red', 'orange', 'purple']

    #plt.figure(0)
    #plt.title('t-SNE')

    patch_size = 299

    dataset = GleasonDataset(trainsets[1:], labels=labels, mode='pca', patch_size=patch_size)
    dataloader = DataLoader(dataset, batch_size=10, drop_last=False)

    N = 512
    max_bat = 100
    max_pat = min(max_bat * dataloader.batch_size, len(dataloader.dataset))
    all_embeddings = np.zeros((max_pat, N))
    all_targets = np.zeros(max_pat)
    all_patches = np.zeros((max_pat, 3, patch_size, patch_size))

    for i, (input, targets) in enumerate(dataloader):
        with torch.no_grad():
            X = model(input)

        X_embedded = X  #TSNE(n_components=N, learning_rate='auto', method='exact',
                        #   init = 'random', perplexity = 3).fit_transform(X)

        batch_sz = X_embedded.shape[0]

        all_embeddings[i*batch_sz:(i+1)*batch_sz, :] = X_embedded
        all_targets[i*batch_sz:(i+1)*batch_sz] = targets
        all_patches[i*batch_sz:(i+1)*batch_sz, :, :, :] = input
        print('processed {} out of {}'.format(i, min(len(dataloader), max_bat)))
        if i+1 == max_bat:
            break
        #plt.figure(0)
        #plt.scatter(X_embedded[:, 0], X_embedded[:, 1], marker='x', color=colors[j])

    kmeans = KMeans(n_clusters=args.n_clusters, random_state=0).fit_predict(all_embeddings)
    #gmm = GaussianMixture(n_components=args.n_clusters, random_state=0).fit_predict(all_embeddings)
    tsne = TSNE(n_components=2, learning_rate='auto', method='exact',
                init='random', perplexity=3).fit_transform(all_embeddings)

    plt.figure(0)
    plt.title('t-SNE - labels')
    for i in range(max_pat):
        plt.scatter(tsne[i,0], tsne[i,1], marker='o', color=colors[int(all_targets[i])])

    plt.figure(1)
    plt.title('t-SNE - prediction')

    # majority prediction between malignant and benign
    cmap = get_cmap(args.n_clusters, name='hsv')
    pred = np.zeros_like(all_targets)
    for k in range(args.n_clusters):
        cluster_labels = all_targets[kmeans==k]
        count = np.bincount(np.int64(cluster_labels))
        pred_label = np.argmax(count)
        acc = max(count)/sum(count)
        pred[kmeans==k] = pred_label
        print('Cluster {} label: {}'.format(k, labels[pred_label]))
        print('N samples: {}'.format(sum(count)))
        print('Accuracy: {}'.format(acc))

        cluster = tsne[kmeans==k, :]
        for j in range(cluster.shape[0]):
            if int(cluster_labels[j]) == 0:
                plt.scatter(cluster[j,0], cluster[j,1], marker='x', color=cmap(k))
            else:
                plt.scatter(cluster[j, 0], cluster[j, 1], marker='o', color=cmap(k), facecolor='none', linewidths=1.5)

    #plt.show()

    total_acc = accuracy_score(all_targets, pred)
    print('Total Accuracy: {}'.format(total_acc))

    # display some images from each cluster

    w, h = 6, args.n_clusters
    imfig = plt.figure(99, figsize=(w, h))
    pos = 1
    for k in range(args.n_clusters):
        indexes = np.array(np.where(kmeans == k)).flatten()
        # three random indexes with cluster k
        samples = np.random.choice(indexes, size=w, replace=False)
        #images = all_patches[indexes, :, :, :]

        for img in all_patches[samples, :, :, :]:
            imfig.add_subplot(h, w, pos)
            plt.imshow(np.moveaxis(img.astype('uint8'), source=0, destination=-1))
            plt.axis('off')
            pos += 1

    plt.show()







