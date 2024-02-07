import matplotlib.pyplot as plt
import torch
import os
# Plot the loss values
def plot_train_val_loss(Lake_name, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join('plots', Lake_name + 'plot_train_val_loss'))
    plt.show()

def plot_val_data(Lake_name, eval_inputs, eval_labels, model, device):

    # Set the model back to evaluation mode
    model.eval()

    # Get the predicted values
    with torch.no_grad():
        predicted_outputs = model(eval_inputs.to(device)).cpu().numpy()

    # Plot the original data and predicted data
    plt.figure(figsize=(10, 6))
    plt.plot(eval_labels.numpy() / 100, label='Original Data', linewidth=2)
    plt.plot(predicted_outputs / 100, label='Predicted Data', linestyle='--', linewidth=2)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Original Data vs Predicted Data', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(os.path.join('plots', Lake_name + 'plot_val_data'))
    plt.show()