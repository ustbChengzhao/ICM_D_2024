import torch.nn as nn
from torch.utils.data import  DataLoader
from torch.optim import Adam

from Preprocessing import read_data, get_dataset
from CFG import CFG
from model import Lstm
from Dataset import MyDataset
from train import train
from utils import plot_train_val_loss, plot_val_data
from inference import inference
import matplotlib.pyplot as plt
if __name__ == "__main__":
    # 读取数据
    data = read_data(CFG.file_path, CFG.Lake_name)
    train_inputs, eval_inputs, train_labels, eval_labels = get_dataset(data, CFG.sequence_length)

    # 构建dataloader
    train_dataset = MyDataset(train_inputs, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    eval_dataset = MyDataset(eval_inputs, eval_labels)
    eval_dataloader = DataLoader(eval_dataset, batch_size=CFG.batch_size, shuffle=False)

    # 实例化模型，损失函数，优化器
    model = Lstm(hidden_size=CFG.hidden_size,
                                  num_layers=CFG.num_layers,
                                  output_size=CFG.output_size,
                                  input_size=CFG.input_size).to(CFG.device)

    loss = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=CFG.lr)
    # 训练
    train_losses, val_losses = train(CFG.Lake_name, train_dataloader, eval_dataloader, model, loss, optimizer, CFG.num_epochs, CFG.device)
    plot_train_val_loss(CFG.Lake_name, train_losses, val_losses)
    plot_val_data(CFG.Lake_name, eval_inputs, eval_labels, model, CFG.device)

    # infer
    infer = inference(model, eval_inputs, CFG.infer_length, CFG.device)
    plt.plot(infer)
    plt.show()



