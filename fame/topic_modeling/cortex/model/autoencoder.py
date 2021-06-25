from typing import List

import gc

import torch
import torch.nn


class MLPAutoEncoder(torch.nn.Module):
    """
    Parameters
    ----------
    input_output_dim: `int`, required
        The dimension of the input

    hidden_layers: `List[int]`, required
        The list of hidden layers for the encoder (the inverse will be used for decoding)

    apply_input_batchnorm: `bool`, optional (default=False)
        If set to True, a general batch-norm will be learned and applied on the inputs.
    """
    def __init__(
            self,
            input_output_dim: int,
            hidden_layers: List[int],
            apply_input_batchnorm: bool = False
    ):
        """
        constructor
        """
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

        encoder_module_list += [torch.nn.LayerNorm(hidden_layers[-1])]

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
        """
        Parameters
        ----------
        x: `torch.Tensor`, required
            The `(batch_size, representation_dim`) input elements.

        return_loss: `bool`, optional (default=False)
            If set to `True`, the loss will be returned.

        return_embeddings:`bool`, optional (default=False)
            If set to True, the output will be the bottleneck representations

        Returns
        ----------
        The output isd either the encoded embeddings, decoded reconstruction, or the loss, depending on the parameters.
        """

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
