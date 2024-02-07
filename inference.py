import torch
import matplotlib.pyplot as plt

from model import Lstm
import numpy as np
from CFG import CFG
from Preprocessing import read_data, get_dataset


def inference(model, infer_data, infer_len, device, batch_size=1):
    # model.load_state_dict(torch.load(model_path, map_location=device))
    inputs = torch.Tensor(infer_data[-1]).to(device)
    infer = []
    model.eval()
    with torch.no_grad():
        for i in range(infer_len):
            out = model(inputs)
            infer.append(out.item())
            inputs = torch.concat((inputs[1:].squeeze(-1), torch.Tensor(out).to(CFG.device))).unsqueeze(-1)
    return infer



