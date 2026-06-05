"""
src/ted_model.py

Temporal Emotional Drift (TED) Model

Implements:
- TEDModel (LSTM + Drift Gate)
- NoDriftLSTM (ablation baseline)
- ted_loss()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TEDModel(nn.Module):
    """
    Proposed BRF-TED model.

    Equations:

        h_t  = LSTM(f_t, h_{t-1})

        y_hat_t = softmax(W h_t + b)

        alpha_t = sigmoid(
            W_alpha * |f_t - f_(t-1)| + b_alpha
        )

        y_tilde_t =
            alpha_t * y_hat_t
            + (1 - alpha_t) * y_tilde_(t-1)
    """

    def __init__(
        self,
        input_dim=13,
        hidden_dim=128,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.dropout_out = nn.Dropout(dropout)

        self.fc_out = nn.Linear(
            hidden_dim,
            num_classes
        )

        self.drift_linear = nn.Linear(
            input_dim,
            1,
            bias=True
        )

    def forward(self, x):
        """
        x shape:
            (batch, seq_len, features)

        returns:
            (batch, seq_len, num_classes)
        """

        h_seq, _ = self.lstm(x)

        logits = self.fc_out(
            self.dropout_out(h_seq)
        )

        y_hat = F.softmax(
            logits,
            dim=-1
        )

        delta = torch.zeros_like(x)

        delta[:, 1:] = (
            x[:, 1:] - x[:, :-1]
        ).abs()

        alpha = torch.sigmoid(
            self.drift_linear(delta)
        )

        outputs = [y_hat[:, 0, :]]

        for t in range(1, x.size(1)):
            a = alpha[:, t, :]

            drift_prediction = (
                a * y_hat[:, t, :]
                + (1 - a) * outputs[-1]
            )

            outputs.append(
                drift_prediction
            )

        return torch.stack(
            outputs,
            dim=1
        )

    def predict(self, x):
        """
        Returns class predictions.
        """

        with torch.no_grad():
            probs = self.forward(x)

        return probs.argmax(dim=-1)


class NoDriftLSTM(nn.Module):
    """
    Ablation baseline.

    Removes the drift gate entirely.
    """

    def __init__(
        self,
        input_dim=13,
        hidden_dim=128,
        num_classes=3,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        h_seq, _ = self.lstm(x)

        logits = self.fc(
            self.dropout(h_seq)
        )

        return F.softmax(
            logits,
            dim=-1
        )


def ted_loss(
    y_tilde,
    targets
):
    """
    Negative log likelihood loss
    on drift-smoothed probabilities.

    Parameters
    ----------
    y_tilde :
        (B,T,C)

    targets :
        (B,T)

    Returns
    -------
    torch.Tensor
    """

    B, T, C = y_tilde.shape

    return F.nll_loss(
        torch.log(
            y_tilde.clamp(min=1e-9)
        ).view(B * T, C),

        targets.view(B * T)
    )


def count_parameters(model):
    """
    Utility function.
    """

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


if __name__ == "__main__":

    model = TEDModel()

    x = torch.randn(
        4,
        8,
        13
    )

    out = model(x)

    print("Output shape:", out.shape)
    print(
        "Parameters:",
        count_parameters(model)
    )