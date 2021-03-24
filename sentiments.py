import torch
import torch.nn as nn
import torch.nn.functional as F



class Classification(nn.Module):
    def __init__(self, LENGTH, embedding_size, model):
        super(Classification, self).__init__()
        self.LENGTH = LENGTH
        self.embedding_size = embedding_size

        self.inputl = nn.Linear(self.LENGTH * 30, embedding_size, 0)  # (LENGTH*30, 300)

        # initialization of weights
        torch.nn.init.xavier_normal_(self.inputl.weight)

        self.FC1 = model.FC1.weight  # (300, 19365)
        self.BN1 = nn.BatchNorm1d(19365)

        self.FC2 = model.FC2.weight  # (19365, 300)
        self.BN2 = nn.BatchNorm1d(300)

        self.FC3 = nn.Linear(300, 1000, 0)  # (300, 1000)
        self.BN3 = nn.BatchNorm1d(1000)

        self.FC4 = nn.Linear(1000, 2000, 0)  # (1000, 2000)
        self.BN4 = nn.BatchNorm1d(2000)

        self.FC5 = nn.Linear(2000, 100, 0)  # (2000, 100)
        self.BN5 = nn.BatchNorm1d(100)

        self.FC6 = nn.Linear(100, 1, 0)  # (100, 1)

        # initialization of weights
        # torch.nn.init.xavier_normal_(self.output.weight)
        self.drop1 = nn.Dropout2d(p=0.8)
        self.drop2 = nn.Dropout2d(p=0.5)

    def forward(self, one_hot):  # one_hot: (64, 30, LENGTH)

        # print(one_hot.shape)
        one_hot = one_hot.reshape(len(one_hot), 30 * self.LENGTH)

        x = self.inputl(one_hot)
        x = F.relu(x)

        x = torch.matmul(x, self.FC1)
        x = self.BN1(x)
        x = F.relu(x)

        x = torch.matmul(x, self.FC2)
        x = self.BN2(x)
        x = F.relu(x)

        x = self.FC3(x)
        x = self.BN3(x)
        x = F.relu(x)
        x = self.drop1(x)

        x = self.FC4(x)
        x = self.BN4(x)
        x = F.relu(x)

        x = self.FC5(x)
        x = self.BN5(x)
        x = F.relu(x)
        x = self.drop2(x)

        x = self.FC6(x)

        m = nn.Sigmoid()
        x = m(x)

        return x

def splitting(df, x, y, z):
  data = df[x:y ]
  if 'category' in data.columns:
    del data['category']

  d2 = df[:z]
  if 'category' in d2.columns:
    del d2['category']

  data = data.append(d2)

  data = data.sample(frac=1)

  data = data.reset_index()

  if 'index' in data.columns:
      del data['index']



  return data

