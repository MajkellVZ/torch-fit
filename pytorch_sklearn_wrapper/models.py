import torch.nn as nn


class SimpleNet(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(SimpleNet, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, x):
		return self.linear(x)


class LSTM(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
		super(LSTM, self).__init__()
		self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
		self.linear = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		out, _ = self.lstm(x)
		out = self.linear(out[:, -1, :])
		return out


class TwoLayerNN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(TwoLayerNN, self).__init__()
		self.layer_1 = nn.Linear(input_dim, hidden_dim)
		nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
		self.layer_2 = nn.Linear(hidden_dim, output_dim)

	def forward(self, x):
		x = nn.functional.relu(self.layer_1(x))
		x = nn.functional.sigmoid(self.layer_2(x))

		return x
