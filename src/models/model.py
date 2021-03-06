import sys

sys.path.append("..")

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
from src.data.prepare_data import MelanomaDataset
from torch.utils.data import DataLoader
from typing import Union
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler


class AverageMeter:
    """
    Keeps track of some summary statistics for the losses when
    training.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_model(model: str, out_features: int):
    """
    Loads the model and sets the correct number of out
    features. Then requires gradients for all but the last 
    layer so that backpropogation can take place
    :param model: name of the neural network architecture to be used
        Here are the choices:
            efficientnet-b0, efficientnet-b1, efficientnet-b2,
            efficientnet-b3, efficientnet-b4, efficientnet-b5,
            efficientnet-b6, efficientnet-b7
            vgg19
            resnext (this is resnext101_32x8d)
            densenet (this is densenet161)
            resnet18 (for local testing)
            alexnet (for local testing)
            
    :param out_features: integer of out features. 2 for Kaggle contest
    """
    net = None
    if model.startswith("efficientnet"):
        net = EfficientNet.from_pretrained(model, num_classes=out_features)
        for name, param in net.named_parameters():
            if "_fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif model == "vgg19":
        net = models.vgg19(pretrained=True)
        net.classifier[6] = nn.Linear(
            in_features=net.classifier[6].weight.size(1), out_features=out_features
        )
        for name, param in net.named_parameters():
            if name in ["classifier.6.weight", "classifier.6.bias"]:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif model == "resnext":
        net = models.resnext101_32x8d(pretrained=True)
        net.fc = nn.Linear(in_features=net.fc.weight.size(1), out_features=out_features)
        for name, param in net.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif model == "densenet":
        net = models.densenet161(pretrained=True)
        net.classifier = nn.Linear(
            in_features=net.classifier.weight.size(1), out_features=out_features
        )
        for name, param in net.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # The next two are small networks for running locally. All
    # others are too large to run locally unless image size is very small

    elif model == "resnet18":
        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(in_features=net.fc.weight.size(1), out_features=out_features)
        for name, param in net.named_parameters():
            if "fc" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif model == "alexnet":
        net = models.alexnet(pretrained=True)
        for name, param in net.named_parameters():
            if "classifier.6" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return net


def eval_model(
    model: torch.nn,
    dataset: MelanomaDataset,
    batch_size: int,
    criterion: torch.nn,
    num_workers: int,
    device=None,
):
    """
    Evaluates the model by passing through images and finding the loss and 
    AUC_ROC score to assess performance
    """
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    losses = AverageMeter()
    preds = []
    targets = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)

            # forward
            outputs = model(inputs)

            # loss
            loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            # prediction
            preds.append(outputs[:, 1])
            targets.append(labels)

        # auc
        preds = torch.cat(preds, dim=-1).cpu().numpy()
        targets = torch.cat(targets, dim=-1).cpu().numpy()
        auc = roc_auc_score(targets, preds)

    return losses.avg, auc


def train_model(
    model_id: str,
    dataset_train: MelanomaDataset,
    dataset_valid: MelanomaDataset,
    batch_size: int,
    model: torch.nn,
    criterion: torch.nn,
    optimizer: torch.nn,
    scheduler: torch.optim,
    num_epochs: int,
    freezed_epochs: int,
    base_dir: str,
    num_workers: int,
    sampler: torch.utils.data.sampler,
    device: str = Union["cuda", "cpu"],
    early_stopping: int = 3,
) -> None:
    """
    :param model_id: A string naming the model (usually architecture with fold number)
    :param dataset_train:
    :param dataset_valid:
    :param batch_size: number of images to be fed to model at a time
    :param model: Result of load_model()
    :param criterion: the loss function
    :param optimizer:
    :param scheduler: the learning rate schedule
    :param num_epochs: total number of epochs for training
    :param freezed_epochs: how many epochs to train the model with all but the final
        layer frozen. We are using transfer learning so final layers need to be 
        trained specifically to the new task
    :param base_dir: location of the images
    :param num_workers: number of cores to be used for image processing
    :param sampler: how to sample the images to combat imbalance
    :param device: "cpu" or "cuda"
    :param early_stopping: if a better auc_valid score is not seen for this number
        of epochs then training is stopped early
    """

    model.to(device)

    # create a dataloader
    if sampler != None:
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
        )
    else:
        dataloader_train = DataLoader(
            dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    # train
    for epoch in range(1, num_epochs + 1):
        losses = AverageMeter()
        s_time = time.time()

        if epoch == freezed_epochs + 1:
            # unfreeze upstream layers
            model.load_state_dict(
                torch.load(f"../models/state_dict_{model_id}.pt", map_location=device)
            )
            for param in model.parameters():
                param.requires_grad = True
            for g in optimizer.param_groups:
                g["lr"] = 1e-4

        # set the train mode
        model.train()

        for data in tqdm(dataloader_train):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = data["inputs"].to(device)
            labels = data["labels"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), inputs.size(0))

        # set the eval mode
        model.eval()

        # calculate validation loss and auc
        loss_valid, auc_valid = eval_model(
            model, dataset_valid, batch_size, criterion, num_workers, device
        )

        # save the checkpoint
        if epoch == 1 or auc_valid > max_auc:
            saved = True
            max_auc = auc_valid
            torch.save(model.state_dict(), f"../models/state_dict_{model_id}.pt")
            counter = 0
        else:
            saved = False
            counter += 1

        # Save on each epoch with the AUC so that if training fails it can be restarted
        # with the best performing loaded
        torch.save(
            model.state_dict(),
            f"../models/state_dict_{model_id}_epoch:_{epoch}_auc_{round(auc_valid, 2)}.pt",
        )

        # print statistics
        e_time = time.time()
        print(
            f"epoch: {epoch}, loss_train: {losses.avg:.4f}, loss_valid: {loss_valid:.4f}, auc_valid: {auc_valid:.4f}, saved: {saved}, {(e_time - s_time):.4f}sec"
        )

        # no operation in freezed epochs
        if epoch < freezed_epochs + 1:
            counter = 0
        else:
            # step the scheduler
            scheduler.step(auc_valid)

            # early stopping
            if early_stopping != None:
                if counter == early_stopping:
                    break


def predict(dataset, batch_size, model, device=None):
    """
    Passes a batch of images through the network and gets their predictions
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    preds = []
    with torch.no_grad():
        for data in dataloader:
            inputs = data["inputs"].to(device)
            outputs = model(inputs)
            preds.append(outputs)
        preds = torch.cat(preds)

    return preds


def get_predictions(
    dataset: MelanomaDataset,
    batch_size: int,
    model: torch.nn,
    test_time_augmentations: int,
    predictions: pd.DataFrame,
    device: str = None,
):
    """
    Pass the images to be predicted through the network performing the given number
    of test time augmentations (TTA). By augmenting the test images we average the 
    errors in the test predicitons. Large errors get averaged to lower ones
    :param dataset:
    :param batch_size:
    :param model:
    :param test_time_augmentations: integer 1 or above
    :param predicitons: dataframe for which predicitions will be appending to in rows
    :param device: "cpu" or "cuda"
    """
    for _ in tqdm(range(test_time_augmentations)):
        pred_test = predict(dataset, batch_size, model, device)
        pred_test = pd.DataFrame(torch.softmax(pred_test, 1)[:, 1].numpy())
        predictions = pd.concat([predictions, pred_test], axis=1)
    return predictions


def create_weighted_random_sampler(train: pd.DataFrame) -> WeightedRandomSampler:
    """
    Self-explanatory
    """
    class_sample_count = np.array(
        [len(np.where(train["target"] == t)[0]) for t in np.unique(train["target"])]
    )
    weight = 1.0 / class_sample_count
    samples_weight = np.array([weight[t] for t in train["target"]])
    return WeightedRandomSampler(samples_weight, len(samples_weight))


def create_tabular_submission(
    model: RandomizedSearchCV,
    X_test: pd.DataFrame,
    sample_path: str,
    submission_name: str,
) -> None:
    """
    Saves in a csv file the predicted probabilies of positive class
    :param model: the model which is already fit
    :param X_test: dataframe containing patient data already formatted
    :param sample_path: this is where the sample submission provided by 
        Kaggle is located. It is altered for our submission
    :param submission_name: the name to be given the submission file
    """
    submission = pd.read_csv(sample_path)
    predictions = model.predict_proba(X_test)
    submission["target"] = predictions[:, 1]
    submission.to_csv(f"../submissions/{submission_name}.csv", index=False)

