from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import random
import seaborn as sn


# DATA AUGMENTATION
plt.xticks(range(1, 11))

train_loss_ctr_crop = np.load('train_loss_CTR_CROP.npy')
val_loss_ctr_crop = np.load('val_loss_CTR_CROP.npy')
plt.plot(train_loss_ctr_crop)
plt.plot(val_loss_ctr_crop)
plt.legend(['Train', 'Val'])
plt.ylabel('Loss')
plt.xlabel('Num Epochs')
plt.savefig('loss_CTR_CROP.jpg')
plt.clf()

train_acc_ctr_crop = np.load('train_acc_CTR_CROP.npy')
val_acc_ctr_crop = np.load('val_acc_CTR_CROP.npy')
plt.plot(train_acc_ctr_crop)
plt.plot(val_acc_ctr_crop)
plt.legend(['Train', 'Val'])
plt.ylabel('Accuracy')
plt.xlabel('Num Epochs')
plt.savefig('acc_CTR_CROP.jpg')
plt.clf()


# normal training (no data augmentation)
train_loss_normal = np.load('train_loss_normal.npy')
val_loss_normal = np.load('val_loss_normal.npy')
plt.plot(train_loss_normal)
plt.plot(val_loss_normal)
plt.legend(['Train', 'Val'])
plt.ylabel('Loss')
plt.xlabel('Num Epochs')
plt.savefig('loss_normal.jpg')
plt.clf()

train_acc_normal = np.load('train_acc_normal.npy')
train_err_normal = [1-curr_acc for curr_acc in train_acc_normal]
val_acc_normal = np.load('val_acc_normal.npy')
plt.plot(train_acc_normal)
plt.plot(val_acc_normal)
plt.legend(['Train', 'Val'])
plt.ylabel('Accuracy')
plt.xlabel('Num Epochs')
plt.savefig('acc_normal.jpg')
plt.clf()

# log-log scale of error as func of training set size
fracs = [1/16, 1/8, 1/4, 1/2, 1]
ds_acc = np.load('test-acc-ds-size.npy')
ds_test_err = [1 - curr_acc for curr_acc in ds_acc]
ds_train_err = np.load('train-err-ds-size.npy')
plt.loglog(fracs, ds_train_err)
plt.loglog(fracs, ds_test_err)
plt.legend(['Train', 'Test'])
plt.xlabel('Log Fraction of Training Set Used')
plt.ylabel('Log Error')
plt.savefig('loglog-err.jpg')
plt.clf()

# show examples where classifier made mistakes
f, axarr = plt.subplots(3,3)
count = 0
for i in range(3):
    for j in range(3):
        axarr[i, j].imshow(np.squeeze(np.load(f'visualization/mistakes/{count}.npy')))
        count += 1
plt.savefig('Mistakes.jpg')
plt.clf()

cf_matrix = np.load('test-conf-mat.npy')
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in range(10)],
                     columns = [i for i in range(10)])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.title('Confusion Matrix')
plt.xlabel('Digit')
plt.ylabel('Digit')
plt.savefig('output.png')
plt.clf()


test_dataset = datasets.MNIST('./data', train=False, download=False,  # No need to download again
                              transform=transforms.ToTensor())

# select 4 images, find the 8 closest images
test_embeddings = np.load('test-embeddings.npy')
img_indices = random.sample(range(test_embeddings.shape[0]), k=4)
closest_imgs_indices = []
for i in range(len(img_indices)):
    curr_img_vec = test_embeddings[i]
    idx_n_dist = []
    for j in range(test_embeddings.shape[0]):
        idx_n_dist.append((j, np.linalg.norm(curr_img_vec - test_embeddings[j])))
    idx_n_dist.sort(key = lambda x: x[1])
    indices_ = [elem[0] for elem in idx_n_dist[:8]]
    closest_imgs_indices.append(indices_)
f, axarr = plt.subplots(4,8)
for i in range(4):
    similar_set = torch.utils.data.Subset(test_dataset, closest_imgs_indices[i])
    loader = torch.utils.data.DataLoader(similar_set, batch_size=1, shuffle=False)
    for j, data in enumerate(loader):
        img, _ = data
        axarr[i, j].imshow(np.squeeze(img.cpu().detach().numpy()))
plt.savefig('similarity.png')
plt.clf()


# tsne
tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(np.load('test-embeddings.npy'))
test_predictions = np.load('test-predictions.npy')
cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(8,8))
num_categories = 10
for lab in range(num_categories):
    indices = np.squeeze(test_predictions==lab)
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.title('Trained Classifier Embedding Space')
plt.savefig('tsne-viz.png')
plt.clf()
