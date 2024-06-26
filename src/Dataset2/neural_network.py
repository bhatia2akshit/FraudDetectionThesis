from alive_progress import alive_bar
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def make_train_step(model, optimizer, loss_fn):
    def train_step(x, y):
        # make prediction
        yhat = model(x)
        model.train()
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss, yhat

    return train_step


def calculate_accuracy(yhat, y):
    yhat = torch.sigmoid(yhat)
    predicted = (yhat > 0.5).float()
    correct = (predicted == y).float().sum()
    accuracy = correct / y.numel()
    return accuracy


def calculate_accuracy_test(yhat, y):
    yhat = torch.sigmoid(yhat)
    predicted = (yhat > 0.5).float()
    correct = (predicted == y).float().sum()
    accuracy = correct / y.numel()
    return accuracy.item(), correct.item(), y.numel()


def plot_roc_curve(writer, y_true, y_scores, epoch):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)
    writer.add_scalar('AUC', roc_auc, epoch)
    writer.add_pr_curve('ROC', y_true, y_scores, epoch)
    return roc_auc


def predict_test_data(testloader, model, writer, epoch, model_name=None ):
    print(f'prediction started for {model_name}')
    all_preds = []
    all_labels = []
    with torch.no_grad():
        cum_loss = 0
        total_correct = 0
        cum_accuracy = 0
        total_images = 0
        for x_batch, y_batch in testloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
            y_batch = y_batch.to(device)

            # model to eval mode
            model.eval()

            yhat = model(x_batch)

            # Collect predictions and labels
            all_preds.append(torch.sigmoid(yhat).cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

            # Calculate accuracy
            accuracy, correct, total = calculate_accuracy_test(yhat, y_batch)
            total_correct += correct
            total_images += total

            # cum_accuracy += accuracy / len(testloader)
    overall_accuracy = total_correct / total_images
    print(f'Overall accuracy on the test dataset: {overall_accuracy * 100:.2f}%')
    print(f'Total correct predictions: {total_correct} out of {total_images} images')

    # Convert lists to numpy arrays
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels).astype(int)

    # Plot ROC curve
    roc_auc = plot_roc_curve(writer, all_labels, all_preds, epoch)
    print(f'ROC AUC: {roc_auc:.2f}')

    # Print confusion matrix and classification report
    all_preds = (all_preds > 0.5).astype(int)
    conf_matrix=confusion_matrix(all_labels, all_preds)

    with open(f'./confusion_matrix.txt','a') as text:
        text.write(f'confusion matrix for {model_name} is: \n')
        np.savetxt(text, conf_matrix, fmt='%d')
        text.write('\n')

    with open(f'./classification_report.txt','a') as text:
        text.write(f'classification report for {model_name} is: \n')
        text.write(classification_report(all_labels, all_preds))
        text.write('\n')

    print('prediction ended.')


def basic_neural_network(model, trainloader, validloader, testloader, model_name=None, n_epochs=30):
    # Initialize the TensorBoard writer
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.fc.parameters())
    train_step = make_train_step(model, optimizer, loss_fn)
    if model_name is None:
        model_name = model._get_name()
    writer = SummaryWriter(f'runs/dataset1_{model_name}')
    losses = []
    val_losses = []

    epoch_train_losses = []
    epoch_test_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []

    early_stopping_tolerance = 5
    early_stopping_threshold = 0.10

    print('starting the training loop')

    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        for i, data in enumerate(trainloader):  # iterate over batches
            x_batch, y_batch = data
            x_batch = x_batch.to(device)  # move to gpu
            y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
            y_batch = y_batch.to(device)  # move to gpu

            loss, yhat = train_step(x_batch, y_batch)
            epoch_loss += loss / len(trainloader)
            losses.append(loss)

            # Calculate accuracy
            accuracy = calculate_accuracy(yhat, y_batch)
            epoch_accuracy += accuracy / len(trainloader)

        epoch_train_losses.append(epoch_loss)
        epoch_train_accuracies.append(epoch_accuracy)

        # if epoch % 4 == 0:
        #     print('\nEpoch : {}, train loss : {}'.format(epoch + 1, epoch_loss))

        # Log the training loss to TensorBoard
        writer.add_scalar(f'train/loss', epoch_loss, epoch)
        writer.add_scalar(f'train/Accuracy', epoch_accuracy, epoch)

        # validation doesn't require gradients
        with torch.no_grad():
            cum_loss = 0
            cum_accuracy = 0
            for x_batch, y_batch in validloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
                y_batch = y_batch.to(device)

                # model to eval mode
                model.eval()

                yhat = model(x_batch)
                val_loss = loss_fn(yhat, y_batch)
                cum_loss += val_loss.item() / len(validloader)
                val_losses.append(val_loss.item())

                # Calculate accuracy
                accuracy = calculate_accuracy(yhat, y_batch)
                cum_accuracy += accuracy / len(validloader)

            epoch_test_losses.append(cum_loss)  # for every epoch, save the validation loss
            # if epoch % 4 == 0:
            #     print('Epoch : {}, val loss : {}'.format(epoch + 1, cum_loss))

            # Log the validation loss to TensorBoard
            writer.add_scalar(f'validation/loss', cum_loss, epoch)
            writer.add_scalar(f'validation/accuracy', cum_accuracy, epoch)

            best_loss = min(epoch_test_losses)

            # save best model
            if cum_loss <= best_loss:
                best_model_wts = model.state_dict()
                best_epoch = epoch

            # early stopping
            early_stopping_counter = 0
            if cum_loss > best_loss:
                early_stopping_counter += 1

            if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
                print("\nTerminating: early stopping")
                break  # terminate training

    # Close the TensorBoard writer
    writer.close()
    model.load_state_dict(best_model_wts)
    predict_test_data(testloader, model, writer, n_epochs, model_name)
    # saving the model dictionary in results/{model_name}
    torch.save(best_model_wts, f'./results/{model_name}_{best_epoch}')

