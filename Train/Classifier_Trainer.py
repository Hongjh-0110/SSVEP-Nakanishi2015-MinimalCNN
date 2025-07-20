# Designer:Pan YuDong
# Coder:God's hand
# Time:2022/7/4 20:40
import torch
import time
from etc.global_config import config
import numpy as np

def calc_itr(acc, n_class, trial_time):
        if acc <= 0 or n_class <= 1:
            return 0.0
        if acc >= 1:
            acc = 0.9999
        term1 = np.log2(n_class)
        term2 = acc * np.log2(acc) if acc > 0 else 0
        term3 = (1 - acc) * np.log2((1 - acc) / (n_class - 1)) if acc < 1 else 0
        itr = (term1 + term2 + term3) * 60 / trial_time
        return itr

def train_on_batch(num_epochs, train_iter, test_iter, optimizer, criterion, net, device, lr_jitter=False):
    algorithm = config['algorithm']
    if algorithm == "DDGCNN":
        lr_decay_rate = config[algorithm]['lr_decay_rate']
        optim_patience = config[algorithm]['optim_patience']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_rate,
                                                               patience=optim_patience, verbose=True, eps=1e-08)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_iter),
                                                               eta_min=5e-6)

    for epoch in range(num_epochs):
        # ==================================training procedure==========================================================
        net.train()
        sum_loss = 0.0
        sum_acc = 0.0
        for data in train_iter:
            if algorithm == "ConvCA":
                X, temp, y = data
                X = X.type(torch.FloatTensor).to(device)
                temp = temp.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                y_hat = net(X, temp)

            else:
                X, y = data
                X = X.type(torch.FloatTensor).to(device)
                y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                y_hat = net(X)

            loss = criterion(y_hat, y).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_jitter and algorithm != "DDGCNN":
                scheduler.step()
            sum_loss += loss.item() / y.shape[0]
            sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()

        train_loss = sum_loss / len(train_iter)
        train_acc = sum_acc / len(train_iter)
        if lr_jitter and algorithm == "DDGCNN":
            scheduler.step(train_acc)

        # ==================================testing procedure==========================================================
        if epoch == num_epochs - 1:
            net.eval()
            sum_acc = 0.0
            for data in test_iter:
                if algorithm == "ConvCA":
                    X, temp, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    temp = temp.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X, temp)

                else:
                    X, y = data
                    X = X.type(torch.FloatTensor).to(device)
                    y = torch.as_tensor(y.reshape(y.shape[0]), dtype=torch.int64).to(device)
                    y_hat = net(X)

                sum_acc += (y == y_hat.argmax(dim=-1)).float().mean()
            val_acc = sum_acc / len(test_iter)
        print(f"epoch{epoch + 1}, train_loss={train_loss:.3f}, train_acc={train_acc:.3f}")
    print(
        f'training finished at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} with final_valid_acc={val_acc:.3f}')
    # 计算并输出ITR
    Nf = config["data_param"]["Nf"]
    ws = config["data_param"]["ws"]
    itr = calc_itr(float(val_acc), Nf, ws)
    print(f'final_valid_ITR={itr:.2f} bits/min')
    torch.cuda.empty_cache()
    return val_acc.cpu().data
