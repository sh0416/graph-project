import os
import argparse
import linecache
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class TripletDataset(Dataset):
    def __init__(self, fpath):
        self.fpath = fpath
        self.size = sum(1 for _ in open(fpath))

    def __getitem__(self, idx):
        return linecache.getline(self.fpath, idx+1).strip()
    
    def __len__(self):
        return self.size


class TripletEmbeddingModel(nn.Module):
    def __init__(self, n_nodes, n_dims):
        super().__init__()
        self.embed = nn.Embedding(n_nodes, n_dims)
        self.criterion = nn.TripletMarginLoss()

    def forward(self, a, p, n):
        return self.criterion(self.embed(a), self.embed(p), self.embed(n))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dims", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--save_every", type=int, default=4000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TripletDataset(os.path.join("prep", "PAMI", "triplet.csv"))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    model = TripletEmbeddingModel(12617, args.n_dims) 
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter()
    step = 0
    for epoch in range(args.epochs):
        with tqdm(loader, desc="Epoch") as train_tbar:
            for batch in train_tbar:
                batch = torch.from_numpy(np.fromstring(','.join(batch), dtype=np.int64, sep=',')).view(args.batch_size, 3)
                batch = batch.to(device)
                loss = model(batch[:, 0], batch[:, 1], batch[:, 2])
                train_tbar.set_postfix(loss="%.4f" % loss)
                writer.add_scalar('train/loss', loss, step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                if step % args.save_every == 0:
                    torch.save(model.state_dict(), "model_%06d.pt" % step)


if __name__ == "__main__":
    main()
