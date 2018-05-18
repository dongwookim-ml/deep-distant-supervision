import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.w1 = nn.Parameter(torch.randn(input_dim, requires_grad=True))

    def forward(self, input):
        # shape (batch, seq_len, hidden_dim) * (hidden_dim, 1) -> (batch, seq_len, 1)
        attn = torch.einsum('ijk,k->ij', [input, self.w1])
        norm_attn = F.softmax(attn, 1).clone()
        summary = torch.einsum("ijk,ij->ik", [input, norm_attn])
        return summary

class DDS(nn.Module):
    """
    Deep distant supervision model
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_relations):
        super(DDS, self).__init__()
        self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.attn = Attention(hidden_dim * 2)
        self.linear = nn.Linear(hidden_dim * 2, num_relations, bias=True)

    def forward(self, input):
        rnn_output, _ = self.rnn(input)
        sen_embedding = self.attn(rnn_output)
        output = self.linear(sen_embedding)
        return output

input_dim = 4
hidden_dim = 3
num_layers = 2
batch_size = 2
seq_len = 5
num_relations = 52

model = DDS(input_dim, hidden_dim, num_layers, num_relations)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

input = torch.randn(batch_size, seq_len, input_dim, requires_grad=False)
output = model(input)

# random output generation
y1= torch.LongTensor(batch_size, 1).random_() % num_relations
y2= torch.LongTensor(batch_size, 1).random_() % num_relations
y_onehot = torch.empty(batch_size, num_relations, requires_grad=False)
y_onehot.zero_()
y_onehot.scatter_(1, y1, 1)
y_onehot.scatter_(1, y2, 1)

loss = loss_fn(output, y_onehot)
print('Loss', loss)

optimizer.zero_grad()
loss.backward()
optimizer.step()

for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.size())

