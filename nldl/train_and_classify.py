import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import argparse
import os
import numpy as np
from skimage.io import imread
import datetime

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt

from image_loaders import GleasonDataset
from resnet import resnet50, resnet101, resnet34, count_parameters

from torchmetrics import ConfusionMatrix
from prettytable import PrettyTable

parser = argparse.ArgumentParser()

parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--which_resnet", type=str, default="resnet34")
parser.add_argument("--use_fcn", type=bool, default=False)
parser.add_argument("--patch_size", type=int, default=299)
parser.add_argument("--model_path", type=str, default="/home/fi5666wi/Documents/Python/torchProject/nldl/saved_models/"
                                                      "simclr_2023-02-16_ep40_model.pth")
parser.add_argument("--n_clusters", type=int, default=10)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--save_dir", type=str, default="/home/fi5666wi/Documents/Python/torchProject/nldl/results")

args = parser.parse_args()
torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RegressionLayer(nn.Module):
    def __init__(self, num_classes, input_channels=512, use_activation=False, use_softmax=False):
        super().__init__()
        self.fc1 = nn.Linear(input_channels,int(input_channels/2))
        self.fc2 = nn.Linear(int(input_channels/2), num_classes)
        self.use_act = use_activation
        self.use_sm = use_softmax
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        if self.use_act:
            #y = self.relu(x.squeeze())
            y = self.fc1(x.squeeze())
            y = self.relu(y)
            y = self.fc2(y)
        else:
            y = self.fc1(x.squeeze())
            y = self.fc2(y)
        if self.use_sm:
            return self.softmax(y)
        else:
            return y


def get_model(freeze=True):
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
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model.to(device)


def train_classifier(model, data_paths, val_paths, mode, labels, patch_size=299, batch_size=10):
    writer = SummaryWriter(os.path.join(args.save_dir, 'tensorboard'))
    tr_dataset = GleasonDataset(data_paths, mode=mode, labels=labels, patch_size=patch_size)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_dataset = GleasonDataset(val_paths, mode=mode, labels=labels, patch_size=patch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    class_weights = tr_dataset.get_weights()

    classifier = RegressionLayer(num_classes=len(labels), use_activation=True, use_softmax=False)
    count_parameters(classifier)
    classifier = classifier.to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), args.learning_rate, weight_decay=10e-6)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), reduction='sum')

    best_acc = 0.0
    n_iter = 0
    for epoch in range(args.epochs):
        print("=====Training Epoch: %d =====" % epoch)
        for i, (x, y) in enumerate(tr_loader):
            x = x.to(device)
            y = y.to(device)

            #with torch.no_grad():
            f = model(x)
            optimizer.zero_grad()
            z = classifier(f)
            loss = loss_fn(z, y)

            if n_iter % 10 == 0:
                writer.add_scalar('train_loss', loss, global_step=n_iter)
                print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch, i, len(tr_loader),
                                                                  loss=loss.item()))

            loss.backward()
            optimizer.step()
            n_iter += 1
        print("===Validation===")
        val_loss = 0.0
        val_acc = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                f = model(x)
                z = classifier(f)
                val_loss += loss_fn(z,y).item()
                y_pred = torch.argmax(F.softmax(z, dim=1), dim=1)
                val_acc += torch.sum(y_pred == y)/y.size(dim=0)
                count += 1
        val_loss /= count
        val_acc /= count
        print("Val:[{0}] loss: {loss:.4f} accuracy: {acc:.4f}".format(epoch, loss=val_loss, acc=val_acc))
        writer.add_scalar('val_loss', val_loss, global_step=epoch)
        writer.add_scalar('val_acc', val_acc, global_step=epoch)

        if val_acc > best_acc:
            print("New Best Accuracy: {}, Saving model".format(val_acc))
            save_name = "RegrLayer_{}_best.pth".format(datetime.date.today())
            torch.save(classifier.state_dict(), os.path.join(args.save_dir, save_name))

    print("=======Training finished======")
    return classifier

def classify(model, classifier, data_paths, mode, labels, patch_size=299, batch_size=10):
    dataset = GleasonDataset(data_paths, mode=mode, labels=labels, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    #correct_pred = {label: 0 for label in labels}
    #total_pred = {label: 0 for label in labels}
    predictions = torch.zeros(len(dataset))
    targets = torch.zeros(len(dataset))

    with torch.no_grad():
        for i, (x, y_true) in enumerate(loader):
            x = x.to(device)
            z = classifier(model(x))
            y_pred = torch.argmax(F.softmax(z, dim=1), dim=1)
            predictions[i*batch_size: (i+1)*batch_size] = y_pred
            targets[i*batch_size: (i+1)*batch_size] = y_true
            """for label, prediction in zip(y_true, y_pred):
                if label == prediction:
                    correct_pred[labels[label]] += 1
                total_pred[labels[label]] += 1"""

    # print accuracy for each class
    """for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')"""

    confmat = ConfusionMatrix("multiclass", num_classes=4)
    conf = confmat(predictions, targets)
    conf = conf.cpu().numpy()

    conf_table = PrettyTable()
    conf_table.add_column(" ", [name for name in labels])
    for i, name in enumerate(labels):
        conf_table.add_column(name, conf[:, i])
    print(conf_table)

    total_acc = torch.sum(predictions == targets)/predictions.size(dim=0)
    print("Total accuracy: {}".format(total_acc))


if __name__ == "__main__":
    cnn = get_model(freeze=False)
    #cnn = torch.hub.load('pytorch/vision', 'resnet34', pretrained=True)
    #cnn = resnet34(args)

    sz = args.patch_size
    trainsets = ['/home/fi5666wi/Documents/Gleason/Data_1_SUS/20181120_SUS/train_' + str(sz),
                '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Helsingborg/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Linkoping/train_' + str(sz),
                 '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/train_' + str(sz)]

    testsets = ['/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_SUS/test_' + str(sz),
                '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Helsingborg/test_' + str(sz),
                '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Linkoping/test_' + str(sz),
                '/home/fi5666wi/Documents/Gleason/Data_0_SUS/20180911_Rotterdam/test_' + str(sz)]

    bludclut = train_classifier(cnn, trainsets[1:], val_paths=[trainsets[0]], mode='gleason', labels=['benign', 'G3', 'G4', 'G5'])

    classify(cnn, bludclut, testsets, mode='gleason', labels=['benign', 'G3', 'G4', 'G5'])









