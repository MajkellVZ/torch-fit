import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SklearnTorchWrapper(BaseEstimator, ClassifierMixin):
	def __init__(self, model, criterion, optimizer, epochs=10, batch_size=32, lr=0.001, device=None):
		assert isinstance(model, nn.Module), "Model must be a PyTorch model (nn.Module)"
		assert callable(criterion), "Criterion must be a callable loss function"
		assert callable(optimizer), "Optimizer must be a callable optimizer"
		assert isinstance(epochs, int) and epochs > 0, "Epochs must be a positive integer"
		assert isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer"
		assert isinstance(lr, float) and lr > 0, "Learning rate must be a positive float"

		self.model = model
		self.criterion = criterion
		self.optimizer = optimizer
		self.epochs = epochs
		self.batch_size = batch_size
		self.lr = lr
		self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)

		logger.info(f"Model initialized on device: {self.device}")

	def fit(self, X, y):
		assert isinstance(X, (torch.Tensor, list, tuple)), "X should be a tensor, list, or tuple"
		assert isinstance(y, (torch.Tensor, list, tuple)), "y should be a tensor, list, or tuple"
		assert len(X) == len(y), "X and y must have the same length"

		self.model.train()
		X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
		y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)

		dataset = TensorDataset(X_tensor, y_tensor)
		loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

		optimizer = self.optimizer(self.model.parameters(), lr=self.lr)

		logger.info(f"Starting training for {self.epochs} epochs...")

		for epoch in range(self.epochs):
			total_loss = 0
			for X_batch, y_batch in loader:
				optimizer.zero_grad()
				output = self.model(X_batch)
				loss = self.criterion(output, y_batch)
				loss.backward()
				optimizer.step()
				total_loss += loss.item()

			avg_loss = total_loss / len(loader)
			logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")

		logger.info("Training complete")
		return self

	def predict(self, X):
		assert isinstance(X, (torch.Tensor, list, tuple)), "X should be a tensor, list, or tuple"

		self.model.eval()
		X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

		with torch.no_grad():
			output = self.model(X_tensor)
			predicted = torch.argmax(output, dim=1)

		logger.info(f"Predicted {len(predicted)} samples.")
		return predicted.cpu().numpy()

	def score(self, X, y):
		assert len(X) == len(y), "X and y must have the same length for scoring"

		y_pred = self.predict(X)
		accuracy = accuracy_score(y, y_pred)
		logger.info(f"Model accuracy: {accuracy:.4f}")
		return accuracy
