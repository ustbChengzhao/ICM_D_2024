from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, input_sequences, output_sequences):
        self.input_sequences = input_sequences
        self.output_sequences = output_sequences

    def __getitem__(self, index):
        return self.input_sequences[index], self.output_sequences[index]

    def __len__(self):
        return len(self.input_sequences)