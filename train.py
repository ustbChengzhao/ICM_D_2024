import torch
import os
def train(Lake_name, train_dataloader, eval_dataloader, model, loss, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []


    # Set the model to training mode
    model.train()

    # Train the model
    for epoch in range(num_epochs):
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)
            l = loss(targets, outputs.reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            # print("inputs: ", inputs)
            # print("outputs: ", outputs)
        # Print the loss for each epoch
        print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {l.item()}")

        # Set the model back to evaluation mode
        # Set the model to evaluation mode
        model.eval()

        # Initialize the total loss
        total_val_loss = 0

        # Iterate over the validation dataloader
        for val_inputs, val_targets in eval_dataloader:
            val_inputs = val_inputs.to(device)
            val_targets = val_targets.to(device)

            # Forward pass
            val_outputs = model(val_inputs)
            val_loss_value = loss(val_outputs.reshape(-1), val_targets)

            # Accumulate the validation loss
            total_val_loss += val_loss_value.item()

        # Calculate the average validation loss
        avg_val_loss = total_val_loss / len(eval_dataloader)

        # Print the average validation loss
        print("Average Validation Loss:", avg_val_loss)

        train_losses.append(l.item())
        val_losses.append(avg_val_loss)
    torch.save(model.state_dict(), os.path.join("model", Lake_name + ".pt"))
    return train_losses, val_losses