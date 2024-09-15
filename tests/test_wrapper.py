import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_sklearn_wrapper import SklearnTorchWrapper, SimpleNet, LSTM, TwoLayerNN


class TestModels(unittest.TestCase):
	def setUp(self):
		# SimpleNet
		self.model_simple = SimpleNet(input_dim=4, output_dim=3)
		# LSTM
		self.model_lstm = LSTM(input_dim=5, hidden_dim=10, num_layers=1, output_dim=3)
		# Two-Layer Neural Network
		self.model_tlnn = TwoLayerNN(input_dim=2, hidden_dim=10, output_dim=3)

		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam

		# SklearnTorchWrapper for testing
		self.wrapper_simple = SklearnTorchWrapper(self.model_simple, self.criterion, self.optimizer, epochs=1)
		self.wrapper_lstm = SklearnTorchWrapper(self.model_lstm, self.criterion, self.optimizer, epochs=1)
		self.wrapper_tlnn = SklearnTorchWrapper(self.model_tlnn, self.criterion, self.optimizer, epochs=1)

		# Dummy data
		self.X_train = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
		self.y_train = [0, 1]

		self.X_test = [[0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9]]
		self.y_test = [0, 1]

		# For LSTM, create dummy sequential data
		self.X_train_lstm = torch.randn(2, 10, 5)  # (batch_size, seq_length, input_dim)
		self.X_test_lstm = torch.randn(2, 10, 5)

	def test_fit_simple_net(self):
		self.wrapper_simple.fit(self.X_train, self.y_train)

	def test_predict_simple_net(self):
		self.wrapper_simple.fit(self.X_train, self.y_train)
		preds = self.wrapper_simple.predict(self.X_test)
		self.assertEqual(len(preds), len(self.y_test))

	def test_score_simple_net(self):
		self.wrapper_simple.fit(self.X_train, self.y_train)
		score = self.wrapper_simple.score(self.X_test, self.y_test)
		self.assertIsInstance(score, float)

	def test_fit_lstm(self):
		self.wrapper_lstm.fit(self.X_train_lstm, self.y_train)

	def test_predict_lstm(self):
		self.wrapper_lstm.fit(self.X_train_lstm, self.y_train)
		preds = self.wrapper_lstm.predict(self.X_test_lstm)
		self.assertEqual(len(preds), len(self.y_test))

	def test_score_lstm(self):
		self.wrapper_lstm.fit(self.X_train_lstm, self.y_train)
		score = self.wrapper_lstm.score(self.X_test_lstm, self.y_test)
		self.assertIsInstance(score, float)

	def test_fit_two_layer_net(self):
		self.wrapper_simple.fit(self.X_train, self.y_train)

	def test_predict_two_layer_net(self):
		self.wrapper_simple.fit(self.X_train, self.y_train)
		preds = self.wrapper_simple.predict(self.X_test)
		self.assertEqual(len(preds), len(self.y_test))

	def test_score_two_layer_net(self):
		self.wrapper_simple.fit(self.X_train, self.y_train)
		score = self.wrapper_simple.score(self.X_test, self.y_test)
		self.assertIsInstance(score, float)


if __name__ == '__main__':
	unittest.main()
