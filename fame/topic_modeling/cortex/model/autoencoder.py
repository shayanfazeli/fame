from typing import List

import gc

import torch
import torch.nn


class MLPAutoEncoder(torch.nn.Module):
    def __init__(
            self,
            input_output_dim: int,
            hidden_layers: List[int],
            apply_input_batchnorm: bool = True
    ):
        super(MLPAutoEncoder, self).__init__()

        self.add_module("criterion", torch.nn.MSELoss())
        self.apply_input_batchnorm = apply_input_batchnorm

        if self.apply_input_batchnorm:
            self.add_module("input_batchnorm", torch.nn.BatchNorm1d(input_output_dim))

        # - encoding
        encoder_module_list = [
            torch.nn.Linear(input_output_dim, hidden_layers[0]),
            torch.nn.BatchNorm1d(hidden_layers[0]),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5)
        ]

        for k in range(1, len(hidden_layers)):
            encoder_module_list += [
                torch.nn.Linear(hidden_layers[k - 1], hidden_layers[k]),
                torch.nn.BatchNorm1d(hidden_layers[k]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=0.5)
            ]

        self.add_module("encoder", torch.nn.Sequential(*encoder_module_list))

        decoder_module_list = []

        # - decoding
        for k in range(len(hidden_layers) - 1, 0, -1):
            decoder_module_list += [
                torch.nn.Linear(hidden_layers[k], hidden_layers[k - 1]),
                torch.nn.BatchNorm1d(hidden_layers[k - 1]),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=0.5)
            ]

        decoder_module_list += [
            torch.nn.Linear(hidden_layers[0], input_output_dim)
        ]

        self.add_module(
            "decoder",
            torch.nn.Sequential(*decoder_module_list)
        )

    def forward(self, x: torch.Tensor, return_loss: bool = False, return_embeddings: bool = False) -> torch.Tensor:
        if self.apply_input_batchnorm:
            x = self.input_batchnorm(x)

        embeddings = self.encoder(x)
        if return_embeddings:
            assert not return_loss
            return embeddings
        else:
            x_hat = self.decoder(embeddings)
            if return_loss:
                return self.criterion(x_hat, x)
            else:
                return x_hat
