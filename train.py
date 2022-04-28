import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import random
from models import Net


### MODEL
model = Net()

### DATASETS
train_dataset = datasets.MNIST('./data', train=True, download=True,  # Downloads into a directory ../data
                                transform=transforms.ToTensor())
                               # transform=transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(24)]))
                               # transform=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(15)]))

# split training dataset into training and validation
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

label_2_dec = dict(zip([625, 125, 25, 50, 100], [1/16, 1/8, 1/4, 1/2, 1]))
train_err_ds_size = []
for ds_frac in [625, 125, 25, 50, 100]:  # frac = [1/16, 1/8, 1/4, 1/2, 1]:
#for ds_frac in [1]:
    print(f'FRACTION OF TRAINING SET USED: {ds_frac}')
    subset_indices = random.sample(range(50000), int(50000*label_2_dec[ds_frac]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, sampler=torch.utils.data.sampler.SubsetRandomSampler(subset_indices))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Some layers, such as Dropout, behave differently during training
    # Train the model for 10 epochs, iterating on the data in batches
    n_epochs = 10

    # store metrics
    training_accuracy_history = np.zeros([n_epochs, 1])
    training_loss_history = np.zeros([n_epochs, 1])
    validation_accuracy_history = np.zeros([n_epochs, 1])
    validation_loss_history = np.zeros([n_epochs, 1])

    last_train_err = None
    for epoch in range(n_epochs):
        print(f'Epoch {epoch+1}/10:', end='')
        train_total = 0
        train_correct = 0
        # train
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            optimizer.zero_grad()
            # forward pass
            output = model(images)
            # calculate categorical cross entropy loss
            loss = loss_fn(output, labels)
            # backward pass
            loss.backward()
            optimizer.step()
            
            # track training accuracy
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            # track training loss
            training_loss_history[epoch] += loss.item()
            # progress update after 180 batches (~1/10 epoch for batch size 32)
            if i % 180 == 0: print('.',end='')
        training_loss_history[epoch] /= len(train_loader)
        training_accuracy_history[epoch] = train_correct / train_total
        print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}, acc: {training_accuracy_history[epoch,0]:0.4f}',end='')
            
        # validate
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(val_loader):
                images, labels = data
                # forward pass
                output = model(images)
                # find accuracy
                _, predicted = torch.max(output.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                # find loss
                loss = loss_fn(output, labels)
                validation_loss_history[epoch] += loss.item()
            validation_loss_history[epoch] /= len(val_loader)
            validation_accuracy_history[epoch] = test_correct / test_total
            last_train_err = 1 - (test_correct / test_total)
        print(f', val loss: {validation_loss_history[epoch,0]:0.4f}, val acc: {validation_accuracy_history[epoch,0]:0.4f}')
    
    torch.save(model.state_dict(), f"my_model_{ds_frac}.pt")

    # UNCOMMENT WHEN RUNNING DATA AUGMENTATION EXPERIMENTS
    # np.save('val_loss_CTR_CROP.npy', validation_loss_history)
    # np.save('val_acc_CTR_CROP.npy', validation_accuracy_history)
    # np.save('train_loss_CTR_CROP.npy', training_loss_history)
    # np.save('train_acc_CTR_CROP.npy', training_accuracy_history)


    train_err_ds_size.append(last_train_err)

np.save('train-err-ds-size.npy', train_err_ds_size)
