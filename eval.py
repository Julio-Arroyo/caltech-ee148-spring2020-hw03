from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import torch.nn as nn
import numpy as np
import torch
from models import Net


test_dataset = datasets.MNIST('./data', train=False, download=False,  # No need to download again
                              transform=transforms.ToTensor())

model = Net()
loss_fn = nn.CrossEntropyLoss()

DONE_TWEAKING = True

test_loss_ds_size = []
test_acc_ds_size = []
for ds_frac in [625, 125, 25, 50, 100]:  # frac in [1/16, 1/8, 1/4, 1/2, 1]:
# for ds_frac in [100]:
    model_pth = f"my_model_{ds_frac}.pt"
    model.load_state_dict(torch.load(model_pth))
    if DONE_TWEAKING:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        # Putting layers like Dropout into evaluation mode
        model.eval()

        test_loss = 0
        correct = 0
        count_mistakes = 0

        y_true_CF = []
        y_pred_CF = []

        test_embeddings = np.zeros((len(test_loader), 100))
        test_predictions = np.zeros((len(test_loader), 1))
        idx = 0
        # Turning off automatic differentiation
        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += loss_fn(output, target).item()  # Sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
                result = pred.eq(target.view_as(pred)).sum().item()

                # confusion matrix
                y_true_CF.append(target.item())
                y_pred_CF.append(pred.item())

                if result == 0 and count_mistakes <= 9:  # save examples
                    np.save(f'visualization/mistakes/{count_mistakes}.npy', data.cpu().detach().numpy())
                    count_mistakes += 1
                correct += result

                # for tsne
                test_embeddings[idx] = model.extract_features(data)
                test_predictions[idx] = pred
                idx += 1

        test_loss /= len(test_loader.dataset)

        print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
            (test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        test_loss_ds_size.append(test_loss)
        test_acc_ds_size.append(correct / len(test_loader.dataset))

        test_conf_mat = confusion_matrix(y_true_CF, y_pred_CF)
        np.save('test-conf-mat.npy', test_conf_mat)
        np.save('test-embeddings.npy', test_embeddings)
        np.save('test-predictions.npy', test_predictions)

np.save('test-loss-ds-size.npy', test_loss_ds_size)
np.save('test-acc-ds-size.npy', test_acc_ds_size)

# if DONE_TWEAKING:
#     # Putting layers like Dropout into evaluation mode
#     model.eval()

#     test_loss = 0
#     correct = 0
#     count_mistakes = 0

#     y_true_CF = []
#     y_pred_CF = []
#     # Turning off automatic differentiation
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             test_loss += loss_fn(output, target).item()  # Sum up batch loss
#             pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max class score
#             result = pred.eq(target.view_as(pred)).sum().item()

#             # confusion matrix
#             y_true_CF.append(target.item())
#             y_pred_CF.append(pred.item())

#             if not result:
#                 np.save(f'visualization/mistakes/{data.cpu().detach().numpy()}.npy')
#                 count_mistakes += 1
#             correct += result

#     test_loss /= len(test_loader.dataset)

#     print('Test set: Average loss: %.4f, Accuracy: %d/%d (%.4f)' %
#         (test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#     test_loss_ds_size.append(test_loss)
#     test_acc_ds_size.append(correct / len(test_loader.dataset))