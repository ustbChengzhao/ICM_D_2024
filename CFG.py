import torch
import os
class CFG:
    file_path = os.path.join("2024_MCM-ICM_Problems", "GLHYD_data_english.csv")
    Lake_name = "Ontario"
    model_path = os.path.join("model", Lake_name + ".pt")
    device = torch.device("mps")
    input_size = 1
    hidden_size = 256
    num_layers = 2
    output_size = 1
    sequence_length = 10
    infer_length = 365
    num_epochs = 25
    batch_size = 32
    lr = 0.00005