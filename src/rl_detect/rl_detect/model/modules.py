"""Modules for the trajectory prediction models."""

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
import lightning as L
import numpy as np
import sklearn

import rl_detect.model.datasets.traj_utils as traj_utils
import rl_detect.model.model_utils as model_utils


"""
Tensor shapes:
- B: batch size
- O: observation length
- o: observation length - 1
- P: prediction length
- A: observation length + prediction length
- 2: x and y coordinates
- H: hidden size
- S: scene size
- s: Max scene size in the batch (to align with other scenes in the batch)
- V: visible scene size
- N: number of negative samples
- C: size of contrastive embedding
- L: latent noise size
- K: number of samples
- E: number of map borders (edges) points
"""


# TODO: maybe split encoders, decoders and social modules in different files?

class TrajectoryEncoderRNN1(nn.Module):
    """Encodes the input trajectory using an RNN.

    Encodes the trajectory using the relative positions between
    consecutive observations.

    The last hidden state is used as the encoding.
    It also returns the last cell state.
    """

    def __init__(self,
                 arch: Literal['gru', 'lstm'],
                 weights_init: Literal['default', 'custom'],
                 hidden_size: int):

        super().__init__()

        input_coords_size = 2

        self.arch = arch
        if self.arch == 'gru':
            rnn_class = nn.GRU
        elif self.arch == 'lstm':
            rnn_class = nn.LSTM
        else:
            raise ValueError(f"Invalid architecture: {arch}")

        self.rnn = rnn_class(input_coords_size,
                             hidden_size,
                             num_layers=1,
                             batch_first=True)

        if weights_init == 'custom':
            model_utils.init_weights(self.arch, self.rnn)

    def forward(self, traj_BO2: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encodes the input trajectory.

        Args:
            scene_SO2: The scene tensor containing the trajectories
                in absolute coordinates.
                The sequence length (obs_len) must be at least 2.
                All the people must be visible, that is the last observation
                must not be nan.
                Shape: (scene_size, obs_len, 2).

        Returns:
            The encoded trajectory tensor (shape: (batch_size, hidden_size)).
        """

        # # Check that all the sequences have at least 2 observations.
        # if not traj_utils.check_enough_observations(traj_BO2).all():
        #     raise ValueError("At least 2 observations per person are needed.")

        # # Check that all the people are visible.
        # if not traj_utils.check_visible(traj_BO2).all():
        #     raise ValueError("All the people must be visible.")

        # Compute relative trajectory.
        traj_rel_Bo2 = traj_BO2.diff(1, dim=1)

        # Encode trajectory.
        # if self.arch == 'gru':
        #     _, hidden_1BH = self.rnn(traj_rel_Bo2)
        #     return hidden_1BH.squeeze(0), None
        # elif self.arch == 'lstm':
        #     _, (hidden_1BH, cell_1BH) = self.rnn(traj_rel_Bo2)
        #     return hidden_1BH.squeeze(0), cell_1BH.squeeze(0)

        # Encode each time step separately in a batched way.
        # This allows to handle sequences that start at different times.

        # Find the first time step with at least one observation.
        first_time_step_idx = 0
        for i in range(traj_rel_Bo2.shape[1]):
            if not torch.isnan(traj_rel_Bo2[:, i]).all():
                # If at least one point is not nan
                # it's the first time step.
                first_time_step_idx = i
                break
        if first_time_step_idx >= traj_rel_Bo2.shape[1]:
            # If all the points are nan throw an error.
            raise ValueError("All the observations are nan.")

        # Initialize hidden state.
        if self.arch == 'gru':
            rnn_hidden_1BH = torch.zeros(1,
                                         traj_rel_Bo2.shape[0],
                                         self.rnn.hidden_size,
                                         device=traj_rel_Bo2.device)
        elif self.arch == 'lstm':
            rnn_hidden_1BH = (torch.zeros(1,
                                          traj_rel_Bo2.shape[0],
                                          self.rnn.hidden_size,
                                          device=traj_rel_Bo2.device),
                              torch.zeros(1,
                                          traj_rel_Bo2.shape[0],
                                          self.rnn.hidden_size,
                                          device=traj_rel_Bo2.device))

        # Encode each time step.
        for i in range(first_time_step_idx, traj_rel_Bo2.shape[1]):
            # Mask for non nan observations.
            non_nan_mask_B = (~torch.isnan(traj_rel_Bo2[:, i])).all(dim=-1)

            # Get current RNN input (non nan observations only).
            curr_input_V12 = traj_rel_Bo2[non_nan_mask_B, i].unsqueeze(1)
            # Get current hidden state (non nan observations only).
            if self.arch == 'gru':
                curr_hidden_1VH = rnn_hidden_1BH[:, non_nan_mask_B] # type: ignore
            elif self.arch == 'lstm':
                curr_hidden_1VH = (rnn_hidden_1BH[0][:, non_nan_mask_B], # type: ignore
                                   rnn_hidden_1BH[1][:, non_nan_mask_B]) # type: ignore

            # Encode time step.
            _, curr_hidden_1VH = self.rnn(curr_input_V12, curr_hidden_1VH) # type: ignore

            # Update hidden state (non nan observations only).
            if self.arch == 'gru':
                rnn_hidden_1BH[:, non_nan_mask_B] = curr_hidden_1VH # type: ignore
            elif self.arch == 'lstm':
                rnn_hidden_1BH[0][:, non_nan_mask_B] = curr_hidden_1VH[0] # type: ignore
                rnn_hidden_1BH[1][:, non_nan_mask_B] = curr_hidden_1VH[1] # type: ignore

        # Get last hidden state.
        if self.arch == 'gru':
            return rnn_hidden_1BH.squeeze(0), None # type: ignore
        elif self.arch == 'lstm':
            return rnn_hidden_1BH[0].squeeze(0), rnn_hidden_1BH[1].squeeze(0) # type: ignore


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float,
                 max_len: int):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0)
                             / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)


class TrajectoryEncoderTransformer1(nn.Module):
    def __init__(self,
                 obs_len: int,
                 num_layers: int,
                 hidden_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 norm_first: bool,
                 activation):

        super().__init__()

        input_coords_size = 2

        self.embedding = nn.Linear(input_coords_size, hidden_size)

        # Max length of positional encoding is the observation length,
        # since the transformer will be applied to the "relative" trajectory,
        # which has length obs_len - 1. And then the additional cls token
        # will be added. So the total length is obs_len.
        self.positional_encoding = PositionalEncoding(hidden_size,
                                                      dropout,
                                                      obs_len)

        self.cls_token_11H = nn.Parameter(torch.zeros(1, 1, hidden_size))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            # TODO: norm ?
        )

    def forward(self, scene_SO2: torch.Tensor) -> torch.Tensor:
        """Encodes the input trajectory.

        Args:
            scene_SO2: The scene tensor containing the trajectories
                in absolute coordinates.
                The sequence length (obs_len) must be at least 2.
                All the people must be visible, that is the last observation
                must not be nan.
                Shape: (scene_size, obs_len, 2).

        Returns:
            The encoded trajectory tensor (shape: (batch_size, hidden_size)).
        """

        # Check that all the sequences have at least 2 observations.
        if not traj_utils.check_enough_observations(scene_SO2).all():
            raise ValueError("At least 2 observations per person are needed.")

        # Check that all the people are visible.
        if not traj_utils.check_visible(scene_SO2).all():
            raise ValueError("All the people must be visible.")

        # Compute relative trajectory.
        scene_rel_So2 = scene_SO2.diff(1, dim=1)

        # Mask for non nan observations.
        # The embedding of every observation is either nan for all
        # the embedding size or not nan for all the embedding size.
        # So it's enough to check only the first entry of the last dimension.
        nan_mask_So = torch.isnan(scene_rel_So2[:, :, 0])

        scene_rel_So2[nan_mask_So] = 0

        # Embed trajectory observations.
        embedded_scene_rel_SoH = self.embedding(scene_rel_So2)

        # Add cls token.
        cls_token_S1H = \
            self.cls_token_11H.expand(embedded_scene_rel_SoH.shape[0], -1, -1)
        input_SOH = torch.cat([cls_token_S1H, embedded_scene_rel_SoH], dim=1)
        nan_mask_SO = torch.cat([torch.ones(nan_mask_So.shape[0],
                                            1,
                                            dtype=bool,
                                            device=nan_mask_So.device),
                                 nan_mask_So],
                                dim=1)

        # Add positional encoding.
        input_SOH = self.positional_encoding(input_SOH)

        # Forward through transformer.
        # Mask padding tokens (True for padding tokens).
        output_SOH = self.transformer_encoder(input_SOH,
                                              src_key_padding_mask=nan_mask_SO)

        # Extract cls token.
        traj_encoding_SH = output_SOH[:, 0, :]

        return traj_encoding_SH, output_SOH



class TrajectoryEncoderTransformer2(nn.Module):
    def __init__(self,
                 obs_len: int,
                 num_layers: int,
                 hidden_size: int,
                 nhead: int,
                 dim_feedforward: int,
                 dropout: float,
                 norm_first: bool,
                 activation):

        super().__init__()

        input_coords_size = 2

        self.embedding = nn.Linear(input_coords_size, hidden_size)

        # Max length of positional encoding is the observation length,
        # since the transformer will be applied to the "relative" trajectory,
        # which has length obs_len - 1. And then the additional cls token
        # will be added. So the total length is obs_len.
        self.positional_encoding = PositionalEncoding(hidden_size,
                                                      dropout,
                                                      obs_len-1) # since relative trajectories

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            # TODO: norm ?
        )

    def forward(self, traj_BO2: torch.Tensor) -> torch.Tensor:
        """Encodes the input trajectory.

        Args:
            scene_SO2: The scene tensor containing the trajectories
                in absolute coordinates.
                The sequence length (obs_len) must be at least 2.
                All the people must be visible, that is the last observation
                must not be nan.
                Shape: (scene_size, obs_len, 2).

        Returns:
            The encoded trajectory tensor (shape: (batch_size, hidden_size)).
        """

        # Compute relative trajectory.
        traj_rel_Bo2 = traj_BO2.diff(1, dim=1)

        # Mask for non nan observations.
        # The embedding of every observation is either nan for all
        # the embedding size or not nan for all the embedding size.
        # So it's enough to check only the first entry of the last dimension.
        nan_mask_Bo = torch.isnan(traj_rel_Bo2[:, :, 0])

        traj_rel_Bo2[nan_mask_Bo] = 0

        # Embed trajectory observations.
        embedded_scene_rel_BoH = self.embedding(traj_rel_Bo2)

        input_BoH = embedded_scene_rel_BoH

        # Add positional encoding.
        input_BoH = self.positional_encoding(input_BoH)

        # Forward through transformer.
        # Mask padding tokens (True for padding tokens).
        output_SoH = self.transformer_encoder(input_BoH,
                                              src_key_padding_mask=nan_mask_Bo)

        return output_SoH


class TrajectoryTokenizer(nn.Module):
    def __init__(self, vocab_size: int, vocab_path: str):
        super().__init__()

        # self.kmeans = sklearn.cluster.KMeans(n_clusters=vocab_size)
        self.vocab_size = vocab_size

        # Load vocabulary.
        cluster_centers_A2 = torch.load(vocab_path)

        self.register_buffer('cluster_centers_A2', cluster_centers_A2)

        # TODO: think how to store precomputed cluster centers


    # def fit(self, traj_BT2: Tensor):
    #     """Fits the tokenizer on the trajectory.

    #     Args:
    #         traj_BT2: The trajectory tensor.
    #             Shape: (batch_size, obs_len, 2).
    #     """

    #     # Compute relative trajectories.
    #     traj_rel_BT2 = traj_BT2.diff(1, dim=1)

    #     # Flatten the trajectories.
    #     traj_rel_B2 = traj_rel_BT2.view(-1, 2)

    #     # To numpy.
    #     traj_rel_B2 = traj_rel_B2.cpu().numpy()

    #     # Fit kmeans.
    #     self.kmeans.fit(traj_rel_B2)

    #     # Get cluster centers.
    #     cluster_centers_A2 = self.kmeans.cluster_centers_
    #     cluster_centers_A2 = torch.from_numpy(cluster_centers_A2).to(traj_BT2.device)
    #     self.cluster_centers_A2 = cluster_centers_A2

    # TODO: implement (encode_smart, decode_predicted)

    def encode(self, traj_BT2: Tensor) -> Tensor:
        """Tokenizes the trajectory.

        Args:
            traj_BO2: The trajectory tensor.
                Shape: (batch_size, obs_len, 2).

        Returns:
            The tokenized trajectory tensor.
            Shape: (batch_size, obs_len).
        """

        batch_size, _, _ = traj_BT2.shape

        # Compute relative trajectories.
        traj_rel_BT2 = traj_BT2.diff(1, dim=1)

        # Compute distance to cluster centers.
        cluster_centers_BA2 = self.cluster_centers_A2.unsqueeze(0).expand(batch_size, -1, -1)
        dist_BTA = torch.cdist(traj_rel_BT2, cluster_centers_BA2, p=2)
        # Token id is the index of the closest cluster center.
        traj_tokens_BT = dist_BTA.argmin(dim=-1)

        return traj_tokens_BT

    def decode(self, traj_tokens_BT: Tensor, first_pos_B2: Tensor) -> Tensor:
        """Decodes the tokenized trajectory.

        Args:
            traj_tokens_BT: The tokenized trajectory tensor.
                Shape: (batch_size, obs_len).

        Returns:
            The decoded trajectory tensor.
            Shape: (batch_size, obs_len, 2).
        """

        # TODO: reconstruct from last observation
        # TODO: reconstruct from last observation
        # TODO: reconstruct from last observation
        # TODO: reconstruct from last observation
        # TODO: reconstruct from last observation
        # TODO: reconstruct from last observation

        # Get cluster centers.
        cluster_centers_A2 = self.cluster_centers_A2

        # Get token values.
        token_values_BT2 = cluster_centers_A2[traj_tokens_BT]

        # Compute absolute trajectory.
        traj_BT2 = torch.cat([first_pos_B2.unsqueeze(1), token_values_BT2], dim=1).cumsum(dim=1)

        return traj_BT2


class TrajectorySampler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                logits_BA: Tensor,
                num_samples: int,
                temperature: float = 1.0,
                top_p: float = 1.0,
                top_k: int = 0,
                min_p: float = 0.0,
                greedy: bool = False) -> Tensor:

        if greedy and num_samples > 1:
            raise ValueError("Greedy sampling only supports num_samples=1.")

        # Sample the next token.
        if greedy:
            return logits_BA.argmax(dim=-1)

        probs_BA = F.softmax(logits_BA / temperature, dim=-1)

        # TODO: check implementation of top_p, top_k, min_p

        # Top-p sampling.
        if top_p < 1.0:
            probs_BA = self._top_p_filter(probs_BA, top_p)

        # Top-k sampling.
        if top_k > 0:
            probs_BA = self._top_k_filter(probs_BA, top_k)

        # Min-p sampling.
        if min_p > 0.0:
            probs_BA = self._min_p_filter(probs_BA, min_p)

        # Sample.
        sample_BK = torch.multinomial(probs_BA, num_samples, replacement=True)

        return sample_BK


    @staticmethod
    def _top_p_filter(probs_BA: Tensor, top_p: float) -> Tensor:
        """Filters the probabilities using top-p sampling.

        Args:
            probs_BA: The probabilities tensor.
                Shape: (batch_size, vocab_size).
            top_p: The top-p threshold.

        Returns:
            The filtered probabilities tensor.
            Shape: (batch_size, vocab_size).
        """

        sorted_probs_BA, sorted_indices_BA = probs_BA.sort(dim=-1, descending=True)

        cum_probs_BA = sorted_probs_BA.cumsum(dim=-1)

        mask_BA = cum_probs_BA <= top_p

        mask_BA[:, -1] = True

        probs_BA = torch.zeros_like(probs_BA)
        probs_BA.scatter_(1, sorted_indices_BA, mask_BA.float())

        return probs_BA

    @staticmethod
    def _top_k_filter(probs_BA: Tensor, top_k: int) -> Tensor:
        """Filters the probabilities using top-k sampling.

        Args:
            probs_BA: The probabilities tensor.
                Shape: (batch_size, vocab_size).
            top_k: The top-k threshold.

        Returns:
            The filtered probabilities tensor.
            Shape: (batch_size, vocab_size).
        """

        sorted_probs_BA, sorted_indices_BA = probs_BA.sort(dim=-1, descending=True)

        mask_BA = torch.zeros_like(probs_BA, dtype=torch.bool)

        mask_BA.scatter_(1, sorted_indices_BA[:, :top_k], True)

        probs_BA = probs_BA.masked_fill(~mask_BA, 0)

        return probs_BA

    @staticmethod
    def _min_p_filter(probs_BA: Tensor, min_p: float) -> Tensor:
        """Filters the probabilities using min-p sampling.

        Args:
            probs_BA: The probabilities tensor.
                Shape: (batch_size, vocab_size).
            min_p: The min-p value.

        Returns:
            The filtered probabilities tensor.
            Shape: (batch_size, vocab_size).
        """

        highest_probs_BA, _ = probs_BA.max(dim=-1)

        min_prob = highest_probs_BA * min_p

        mask_BA = probs_BA >= min_prob.unsqueeze(-1)

        probs_BA = probs_BA.masked_fill(~mask_BA, 0)

        return probs_BA




class SWiGLU(nn.Module):
    def __init__(self, d_model):
        super(SWiGLU, self).__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.linear1(x) * F.silu(self.linear2(x))
        # return self.linear1(x) * torch.sigmoid(self.linear2(x))


class SocialEncoder1(nn.Module):
    def __init__(self,
                 social_info_type: Literal["absolute", "relative"],
                 num_layers,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout,
                 norm_first,
                 activation):
        """Builds a social encoder.

        Args:
            social_info_type: The type of social information to use.
                Can be one of:
                - "absoulte": Use the absolute positions of the people.
                - "relative": Use the relative positions of the people
                    between them.
            # TODO

        """

        super().__init__()

        # Validate social info type.
        if social_info_type not in ["absolute", "relative"]:
            raise ValueError("Invalid social info type. Must be one of: "
                             "'absolute', 'relative'.")

        self.social_info_type = social_info_type

        input_coords_size = 2

        self.coord_embedding = nn.Linear(input_coords_size, d_model)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer,
            num_layers=num_layers,
            # TODO: norm ?
        )

        # TODO parametrize social encoder architecture (e.g. number of layers)
        # num_layers = 0: just attention
        # self.transformer_encoder = MyAttention(d_model, nhead, dropout)



    def forward(self,
                traj_BO2: torch.Tensor,
                scene_idx_B: torch.Tensor,
                traj_encoding_BH: torch.Tensor) -> torch.Tensor:

        if self.social_info_type == "absolute":
            return self._forward_absolute(traj_BO2, scene_idx_B, traj_encoding_BH)

        elif self.social_info_type == "relative":
            return self._forward_relative(traj_BO2, scene_idx_B, traj_encoding_BH)

        else:
            # This should never happen.
            assert False, "Invalid social info type."


    def _forward_absolute(self,
                          traj_BO2,
                          scene_idx_B,
                          traj_encoding_BH):
        # Get last absolute position.
        last_pos_B2 = traj_BO2[:, -1, :]
        # Embed last position.
        last_pos_BH = self.coord_embedding(last_pos_B2)

        # Sum hidden state and last position embedding.
        encoding_BH = traj_encoding_BH + last_pos_BH

        # Add batch dimension for transformer.
        traj_encoding_1BH = encoding_BH.unsqueeze(0)

        # Create attention mask: only attend to people in the same scene.
        attn_mask_BB = scene_idx_B.unsqueeze(0) != scene_idx_B.unsqueeze(1)

        # Compute social encoding.
        social_encoding_1BH = self.transformer_encoder(traj_encoding_1BH,
                                                       mask=attn_mask_BB)

        return social_encoding_1BH.squeeze(0)

    # TODO: maybe pass also the current time step (default to None)
    def _forward_relative(self,
                          traj_BO2,
                          scene_idx_B,
                          traj_encoding_BH):

        # Batch size.
        batch_size, _, _ = traj_BO2.shape

        # Last known position.
        last_pos_B2 = traj_BO2[:, -1, :]

        # TODO: in principle, this should be done only for the people in the same scene

        # Compute relative position for each pair of people.
        rel_pos_BB2 = last_pos_B2.unsqueeze(1) - last_pos_B2.unsqueeze(0)

        # Expand trajectory encoding.
        traj_encoding_BBH = \
            traj_encoding_BH.unsqueeze(0).expand(batch_size, -1, -1)

        # Compute relative position embedding.
        rel_pos_embedding_BBH = self.coord_embedding(rel_pos_BB2)

        # Sum relative position embedding and trajectory encoding.
        rel_traj_encoding_BBH = rel_pos_embedding_BBH + traj_encoding_BBH

        # Create attention mask: only attend to people in the same scene.
        attn_mask_BB = scene_idx_B.unsqueeze(0) != scene_idx_B.unsqueeze(1)
        attn_mask_BB = attn_mask_BB.to(device=rel_traj_encoding_BBH.device)
        # Compute social encoding.
        social_encoding_BBH = self.transformer_encoder(rel_traj_encoding_BBH,
                                                       mask=attn_mask_BB)

        # Extract social encoding for each person (diagonal of the matrix).
        social_encoding_BH = torch.diagonal(social_encoding_BBH).T

        return social_encoding_BH.contiguous()


class SocialEncoder2(nn.Module):
    """DirectConcat social encoder.

    Considers only the k closest neighbors to the current person.

    Concatenates relative velocity and relative position (embeddings ?)
    of top-k neighbors.

    Then pass aggregated vector through a LSTM.
    """
    # TODO: eventually merge to SocialEncoder1

    def __init__(self,
                 num_neighbors: int,
                 hidden_size: int,
                 spatial_embedding_size: int,
                 velocity_embedding_size: int):

        super().__init__()

        # TODO think better about the sizes and on how to aggregate different embeddings

        self.num_neighbors = num_neighbors

        self.spatial_embedding_size = spatial_embedding_size
        self.velocity_embedding_size = velocity_embedding_size

        # Embedding for relative position.
        # self.spatial_embedding = nn.Linear(2, spatial_embedding_size)
        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, spatial_embedding_size),
            nn.ReLU()
        )

        # Embedding for relative velocity.
        # self.velocity_embedding = nn.Linear(2, velocity_embedding_size)
        self.velocity_embedding = nn.Sequential(
            nn.Linear(2, velocity_embedding_size),
            nn.ReLU()
        )

        # Aggregate top k neighbors.
        aggregator_input_size = \
            (spatial_embedding_size + velocity_embedding_size) * num_neighbors
        self.aggregator = nn.Linear(aggregator_input_size, hidden_size)

    def forward(self,
                curr_pos_S2: torch.Tensor,
                prev_pos_S2: torch.Tensor,
                traj_encoding_SH: torch.Tensor):
        """Computes the social encoding.

        Args:
            curr_pos_S2: The current position of the people.
                Shape: (scene_size, 2).
            prev_pos_S2: The previous position of the people.
                Shape: (scene_size, 2).
            traj_encoding_SH: The trajectory encoding.
                Shape: (scene_size, hidden_size).
        """

        # K: number of neighbors
        # k: min(K, scene_size)

        scene_size, _ = curr_pos_S2.shape

        # TODO: eventually extract to a separate function

        # Compute relative position for each pair of people.
        rel_pos_SS2 = curr_pos_S2.unsqueeze(1) - curr_pos_S2.unsqueeze(0)

        # Compute velocity of each person.
        vel_S2 = curr_pos_S2 - prev_pos_S2

        # Compute relative velocity for each pair of people.
        rel_vel_SS2 = vel_S2.unsqueeze(1) - vel_S2.unsqueeze(0)

        # Sort by distance, once for each person.
        sorted_indices_SS = torch.argsort(rel_pos_SS2.norm(dim=-1), dim=-1)

        # Get top k neighbors (skip first one, which is the person itself).
        top_k_indices_Sk = sorted_indices_SS[:, 1:self.num_neighbors + 1]

        # Extract top k relative positions and velocities.
        top_k_rel_pos_Sk2 = rel_pos_SS2[torch.arange(scene_size).unsqueeze(1),
                                        top_k_indices_Sk]
        top_k_rel_vel_Sk2 = rel_vel_SS2[torch.arange(scene_size).unsqueeze(1),
                                        top_k_indices_Sk]

        # Embed relative position and velocity.
        top_k_rel_pos_SkH = self.spatial_embedding(top_k_rel_pos_Sk2)
        top_k_rel_vel_SkH = self.velocity_embedding(top_k_rel_vel_Sk2)

        # Pad with zeros if scene size is less than k.
        # Padding size.
        pad_size = max(0, self.num_neighbors - scene_size + 1)
        # Pad with zeros at the end.
        top_k_rel_pos_SKH = \
            torch.cat([top_k_rel_pos_SkH,
                       torch.zeros(scene_size,
                                   pad_size,
                                   self.spatial_embedding_size,
                                   device=top_k_rel_pos_SkH.device)],
                      dim=-2)
        top_k_rel_vel_SKH = \
            torch.cat([top_k_rel_vel_SkH,
                       torch.zeros(scene_size,
                                   pad_size,
                                   self.velocity_embedding_size,
                                   device=top_k_rel_vel_SkH.device)],
                      dim=-2)

        # Concatenate relative position and relative velocity embeddings.
        top_k_rel_pos_vel_SKH = \
            torch.cat([top_k_rel_pos_SKH, top_k_rel_vel_SKH], dim=-1)

        # Aggregate top k neighbors.
        aggregated_SH = \
            self.aggregator(top_k_rel_pos_vel_SKH.view(scene_size, -1))

        # TODO
        # ? pass through lstm: maybe not here, but in decoder
        # ? pass to lstm social state at each step (check paper)
        # ? likely here using lstmCell, and then pass in prev

        # TODO: maybe try an alternative that merges also the trajectory encoding
        # implemented below (think if right thing to do)

        output_SH = traj_encoding_SH + aggregated_SH

        return output_SH


class TrajectoryDecoderRNN0(nn.Module):
    """RNN decoder for trajectory prediction.

    Decodes the trajectory using an RNN (GRU or LSTM).
    Predicts the whole future trajectory at once (as specified by pred_len).
    More general wrt TrajectoryDecoderRNN1.
    """

    def __init__(self,
                 arch: Literal['gru', 'lstm'],
                 weights_init: Literal['default', 'custom'],
                 hidden_size,
                 pred_len):

        super().__init__()

        input_coords_size = 2
        output_coords_size = 2

        self.pred_len = pred_len

        self.arch = arch
        if self.arch == 'gru':
            rnn_class = nn.GRU
        elif self.arch == 'lstm':
            rnn_class = nn.LSTM
        else:
            raise ValueError(f"Invalid architecture: {arch}")

        self.rnn = rnn_class(input_coords_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True)


        if weights_init == 'custom':
            model_utils.init_weights(self.arch, self.rnn)

        self.output = nn.Linear(hidden_size, output_coords_size)

    def forward(self, input_B2, hidden_BH, cell_BH, last_pos_B2):
        batch_size, _ = hidden_BH.shape

        output_BP2 = torch.zeros(batch_size,
                                 self.pred_len,
                                 2,
                                 device=hidden_BH.device)

        # Use last known position as input for first step.
        input_B12 = input_B2.unsqueeze(1)

        # Reshape hidden and cell state for LSTM.
        hidden_1BH = hidden_BH.unsqueeze(0)
        if self.arch == 'lstm':
            cell_1BH = cell_BH.unsqueeze(0)

        # Predict trajectory.
        for i in range(self.pred_len):
            # Decode trajectory.
            if self.arch == 'gru':
                curr_output_B1H, hidden_1BH = self.rnn(input_B12, hidden_1BH)
            elif self.arch == 'lstm':
                curr_output_B1H, (hidden_1BH, cell_1BH) = \
                    self.rnn(input_B12, (hidden_1BH, cell_1BH)) # type: ignore

            output_BP2[:, i, :] = self.output(curr_output_B1H.squeeze(1)) # type: ignore

            # Use last output as input for next step.
            input_B12 = output_BP2[:, i, :].unsqueeze(1).clone()

        # Compute absolute trajectory.
        output_BP2 = output_BP2.cumsum(dim=1) + last_pos_B2.unsqueeze(1)

        return output_BP2


class TrajectoryDecoderRNN1(nn.Module):
    """RNN decoder for trajectory prediction.

    Decodes the trajectory using an RNN (GRU or LSTM).
    Predicts the whole future trajectory at once (as specified by pred_len).
    Relies on TrajectoryDecoderRNN0, with a learned start token as input,
    and a mlp to initialize the cell state for LSTM.
    """
    def __init__(self,
                 arch: Literal['gru', 'lstm'],
                 weights_init: Literal['default', 'custom'],
                 hidden_size,
                 pred_len):

        super().__init__()

        input_size_coords = 2
        # output_coords_size = 2

        self.pred_len = pred_len

        self.arch = arch

        self.decoder = TrajectoryDecoderRNN0(arch, weights_init, hidden_size, pred_len)

        self.start_token_12 = nn.Parameter(torch.zeros(1, input_size_coords))

        if arch == 'lstm':
            self.cell_state_mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

    def forward(self, social_encoding_BH, last_pos_B2):
        scene_size, _ = social_encoding_BH.shape

        output_BP2 = torch.zeros(scene_size,
                                 self.pred_len,
                                 2,
                                 device=social_encoding_BH.device)

        start_token_B2 = \
            self.start_token_12.expand(scene_size, -1)

        # Initialize hidden state.
        hidden_BH = social_encoding_BH
        cell_BH = None
        if self.arch == 'lstm':
            cell_BH = self.cell_state_mlp(social_encoding_BH)

        # Predict trajectory.
        output_BP2 = self.decoder(start_token_B2,
                                  hidden_BH,
                                  cell_BH,
                                  last_pos_B2)

        return output_BP2


# TODO: likely dependency injection for social encoder
class TrajectoryDecoderLSTM3(nn.Module):
    """Decodes trajectories computing the social encoding at each step."""
    def __init__(self,
                 hidden_size,
                 social_encoder_num_layers,
                 nhead,
                 dim_feedforward,
                 dropout,
                 norm_first,
                 activation,
                 pred_len):

        super().__init__()

        input_size_coords = 2
        output_coords_size = 2

        lstm_input_size = hidden_size * 2

        self.pred_len = pred_len

        self.merge_mlp = nn.Sequential(
            nn.Linear(hidden_size + input_size_coords, lstm_input_size),
            nn.ReLU()
        )

        self.lstm = nn.LSTM(lstm_input_size,
                            hidden_size,
                            num_layers=1,
                            batch_first=True)

        self.social_encoder = SocialEncoder1(social_encoder_num_layers,
                                             hidden_size,
                                             nhead,
                                             dim_feedforward,
                                             dropout,
                                             norm_first,
                                             activation)

        # TODO: maybe don't learn this?
        # TODO: or use last encoder output?
        self.start_token_112 = \
            nn.Parameter(torch.zeros(1, 1, input_size_coords))

        self.output = nn.Linear(hidden_size, output_coords_size)


    def forward(self,
                traj_encoding_SH,
                cell_state_SH,
                last_pos_S2):
        scene_size, _ = traj_encoding_SH.shape

        output_SP2 = torch.zeros(scene_size,
                                 self.pred_len,
                                 2,
                                 device=traj_encoding_SH.device)

        start_token_S12 = \
            self.start_token_112.expand(scene_size, -1, -1)

        # Initialize hidden state.
        hidden_1SH = traj_encoding_SH.unsqueeze(0)
        cell_1SH = cell_state_SH.unsqueeze(0)

        # Initialize last known position.
        new_last_pos_S12 = last_pos_S2.unsqueeze(1)

        # Use start token as input for first step.
        input_S12 = start_token_S12

        # Predict trajectory.
        for i in range(self.pred_len):
            # Compute social encoding.
            # Last known (absolute) position.
            social_encoding_SH = self.social_encoder(new_last_pos_S12,
                                                     hidden_1SH.squeeze(0))

            # Save first social encoding.
            if i == 0:
                first_social_encoding_SH = social_encoding_SH

            input_S12 = self.merge_mlp(
                torch.cat([input_S12.squeeze(1), social_encoding_SH], dim=-1)
            ).unsqueeze(1)

            # Decode trajectory.
            curr_output_S1H, (hidden_1SH, cell_1SH) = \
                self.lstm(input_S12, (hidden_1SH, cell_1SH))

            output_SP2[:, i, :] = self.output(curr_output_S1H.squeeze(1))

            # Use last output as input for next step.
            input_S12 = output_SP2[:, i, :].unsqueeze(1).clone()
            # Update last known position.
            new_last_pos_S12 += output_SP2[:, i, :].unsqueeze(1).clone()

        # Compute absolute trajectory.
        output_SP2 = output_SP2.cumsum(dim=1) + last_pos_S2.unsqueeze(1)

        if self.training:
            return output_SP2, first_social_encoding_SH

        return output_SP2


class TrajectoryDecoderTransformer1(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 nhead,
                 dim_feedforward,
                 dropout,
                 norm_first,
                 activation,
                 pred_len):

        super().__init__()

        output_coords_size = 2

        self.pred_len = pred_len

        # Max length of positional encoding is the prediction length,
        # since the transformer will need to predict pred_len steps,

        # TODO: maybe use separate dropout
        self.positional_encoding = PositionalEncoding(hidden_size,
                                                      dropout,
                                                      pred_len + 1)

        # Use decoder-only transformer (GPT style).
        # No encoder-decoder attention, so use transformer encoder
        # with causal mask.
        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=num_layers,
            # TODO: norm ?
        )

        self.output = nn.Linear(hidden_size, output_coords_size)

    def forward(self, social_encoding_SH, memory_SOH, last_pos_S2):
        scene_size, hidden_size = social_encoding_SH.shape

        output_SP2 = torch.zeros(scene_size,
                                 self.pred_len,
                                 2,
                                 device=social_encoding_SH.device)

        # Input for first step.
        input_SPH = torch.full((scene_size, self.pred_len, hidden_size),
                               np.nan,
                               device=social_encoding_SH.device)
        new_input_SH = social_encoding_SH

        # Predict trajectory.
        for i in range(self.pred_len):
            # Build input for current step.
            input_SPH[:, i, :] = new_input_SH

            # Add positional encoding.
            input_SPH = self.positional_encoding(input_SPH)

            # Attention causal mask.
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                self.pred_len,
                device=social_encoding_SH.device
            )
            # attn_mask = attn_mask != torch.inf
            # Mask for non nan observations.
            padding_mask_SP = torch.isnan(input_SPH[:, :, 0])

            true_input_SPH = input_SPH.clone()
            true_input_SPH[padding_mask_SP] = 0

            # Predict trajectory.
            curr_output_SPH = self.transformer_decoder(
                true_input_SPH,
                memory_SOH,
                tgt_mask=attn_mask,
                tgt_key_padding_mask=padding_mask_SP,
                tgt_is_causal=True
            )

            output_SP2[:, i, :] = self.output(curr_output_SPH[:, i, :])

            # Use last output as input for next step.
            new_input_SH = curr_output_SPH[:, i, :]

        # Compute absolute trajectory.
        output_SP2 = output_SP2.cumsum(dim=1) + last_pos_S2.unsqueeze(1)

        return output_SP2




class TrajectoryDecoderTransformer2(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 nhead,
                 dim_feedforward,
                 dropout,
                 norm_first,
                 activation,
                 pred_len,
                 num_samples):

        super().__init__()

        output_coords_size = 2
        self.hidden_size = hidden_size

        self.pred_len = pred_len
        self.num_samples = num_samples


        # Max length of positional encoding is the prediction length,
        # since the transformer will need to predict pred_len steps,

        # TODO: maybe use separate dropout
        # TODO: would like to use 2 separate types of positional encoding_SH
        # (time and samples)
        self.positional_encoding = PositionalEncoding(hidden_size,
                                                      dropout,
                                                      pred_len * num_samples)

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=num_layers,
            # TODO: norm ?
        )

        self.output = nn.Linear(hidden_size, output_coords_size)

    def forward(self, memory_BoH, last_pos_B2):
        batch_size = memory_BoH.shape[0]

        input_BKPH = torch.zeros(batch_size,
                                 self.num_samples,
                                 self.pred_len,
                                 self.hidden_size,
                                 device=memory_BoH.device)

        # Flatten input for transformer (T=K*P).
        input_BTH = input_BKPH.view(batch_size, -1, self.hidden_size)
        # Positional encoding.
        input_BTH = self.positional_encoding(input_BTH)

        # TODO: causal mask ? (likely not)

        # Predict.
        output_BTH = self.transformer_decoder(input_BTH, memory_BoH)

        # Reshape output.
        output_BKPH = output_BTH.view(batch_size,
                                      self.num_samples,
                                      self.pred_len,
                                      self.hidden_size)

        # Project to output space.
        output_BKP2 = self.output(output_BKPH)

        # Compute absolute trajectory.
        output_BKP2 = output_BKP2.cumsum(dim=2) + last_pos_B2[:, None, None, :]

        return output_BKP2


class TrajectoryDecoderTransformer3(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 nhead,
                 dim_feedforward,
                 dropout,
                 norm_first,
                 activation,
                 pred_len,
                 num_samples):

        super().__init__()

        output_coords_size = 2
        self.hidden_size = hidden_size

        self.pred_len = pred_len
        self.num_samples = num_samples


        # Max length of positional encoding is the prediction length,
        # since the transformer will need to predict pred_len steps,

        # TODO: maybe use separate dropout
        # TODO: would like to use 2 separate types of positional encoding_SH
        # (time and samples)
        self.positional_encoding = PositionalEncoding(hidden_size,
                                                      dropout,
                                                      pred_len * num_samples)

        transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            norm_first=norm_first,
            batch_first=True
        )

        self.transformer_decoder = nn.TransformerDecoder(
            transformer_decoder_layer,
            num_layers=num_layers,
            # TODO: norm ?
        )

        self.output = nn.Linear(hidden_size, output_coords_size)

    def forward(self, memory_BOH, last_pos_B2):
        batch_size = memory_BOH.shape[0]

        input_BPH = torch.zeros(batch_size,
                                self.pred_len,
                                self.hidden_size,
                                device=memory_BOH.device)

        # Positional encoding.
        input_BPH = self.positional_encoding(input_BPH)

        # TODO: causal mask ? (likely not)

        # Predict.
        output_BPH = self.transformer_decoder(input_BPH, memory_BOH)

        # Project to output space.
        output_BP2 = self.output(output_BPH)

        # Compute absolute trajectory.
        output_BP2 = output_BP2.cumsum(dim=1) + last_pos_B2[:, None, :]

        return output_BP2
