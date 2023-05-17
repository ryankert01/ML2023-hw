import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter, UninitializedParameter
from conformer.encoder import ConformerBlock

class SelfAttentionPooling(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		self.W = nn.Linear(input_dim, 1)
		self.softmax = nn.Softmax(dim=1)
		
	def forward(self, batch_rep):
		out = self.W(batch_rep).squeeze(-1)
		att_w = self.softmax(out).unsqueeze(-1)
		utter_rep = torch.sum(batch_rep * att_w, dim=1)

		return utter_rep


class Classifier(nn.Module):
	def __init__(self, d_model=80, n_spks=600, dropout=0.1):
		super().__init__()

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		# Project the dimension of features from that of input into d_model.
		self.prenet = nn.Linear(40, d_model)
		# TODO:
		#   Change Transformer to Conformer.
		#   https://arxiv.org/abs/2005.08100
		# self.encoder_layer = nn.TransformerEncoderLayer(
		# 	d_model=d_model, dim_feedforward=256, nhead=1
		# )

		self.conformer1 = ConformerBlock(encoder_dim = d_model,
										num_attention_heads = 8,
										feed_forward_expansion_factor = 4,
										conv_expansion_factor = 2,
										feed_forward_dropout_p = 0.1,
										attention_dropout_p = 0.1,
										conv_dropout_p = 0.1,
										conv_kernel_size = 31,
										half_step_residual = True)
		
		self.conformer2 = ConformerBlock(encoder_dim = d_model,
										num_attention_heads = 8,
										feed_forward_expansion_factor = 4,
										conv_expansion_factor = 2,
										feed_forward_dropout_p = 0.1,
										attention_dropout_p = 0.1,
										conv_dropout_p = 0.1,
										conv_kernel_size = 31,
										half_step_residual = True)
		
		self.conformer3 = ConformerBlock(encoder_dim = d_model,
										num_attention_heads = 8,
										feed_forward_expansion_factor = 4,
										conv_expansion_factor = 2,
										feed_forward_dropout_p = 0.1,
										attention_dropout_p = 0.1,
										conv_dropout_p = 0.1,
										conv_kernel_size = 31,
										half_step_residual = True)
		self.SAPooling = SelfAttentionPooling(input_dim=d_model)
		# self.encoder = nn.TransformerEncoder(self.conformer, num_layers=2)
		# Project the the dimension of features from d_model into speaker nums.
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.Sigmoid(),
			nn.Linear(d_model, n_spks),
		)

	def forward(self, mels):
		"""
		args:
			mels: (batch size, length, 40)
		return:
			out: (batch size, n_spks)
		"""
		# out: (batch size, length, d_model)
		out = self.prenet(mels)
		# out: (length, batch size, d_model)
		# out = out.permute(1, 0, 2)
		# The encoder layer expect features in the shape of (length, batch size, d_model).
		# out = self.encoder(out)
		out = self.conformer1(out)
		out = self.conformer2(out)
		out = self.conformer3(out)
		# out: (batch size, length, d_model)
		stats = self.SAPooling(out)

		# out: (batch, n_spks)
		out = self.pred_layer(stats)
		return out