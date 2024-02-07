import pandas as pd
from sklearn.preprocessing import scale
import torch
import numpy as np
def read_data(file_path, Lake_name):
    data = pd.read_csv(file_path)
    Lake_data = data[Lake_name].values
    Lake_data = scale(Lake_data)
    return Lake_data


def get_dataset(data, sequence_length):
    input_sequences = []
    output_sequences = []

    for i in range(len(data) - sequence_length):
        seq = data[i:i + sequence_length]
        target = data[i + sequence_length]
        input_sequences.append(seq)
        output_sequences.append(target)
    input_sequences = np.array(input_sequences)
    output_sequences = np.array(output_sequences)
    input_sequences = torch.tensor(input_sequences, dtype=torch.float32).unsqueeze(-1)
    output_sequences = torch.tensor(output_sequences, dtype=torch.float32)

    train_inputs = input_sequences[:int(len(input_sequences) * 0.8)]
    eval_inputs = input_sequences[int(len(input_sequences) * 0.8):]
    train_labels = output_sequences[:int(len(output_sequences) * 0.8)]
    eval_labels = output_sequences[int(len(output_sequences) * 0.8):]

    return train_inputs, eval_inputs, train_labels, eval_labels