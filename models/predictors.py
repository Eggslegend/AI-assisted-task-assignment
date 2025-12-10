import torch
import torch.nn as nn

class ScorePredictor(nn.Module):
    """MLP for predicting completion scores."""

    def __init__(self, input_dim, hidden_layers, dropout_rate):
        super(ScorePredictor, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def reset_weights(self):
        for layer in self.network:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

class TransformerScorePredictor(nn.Module):
    """
    Transformer-based model for predicting task completion scores.

    Uses self-attention to capture relationships between:
    - Person characteristics (skills)
    - Task characteristics (type, difficulty)
    - Historical context (experience)
    """

    def __init__(self, input_dim, config):
        super(TransformerScorePredictor, self).__init__()

        self.config = config
        d_model = config['d_model']

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_encoding = nn.Parameter(torch.randn(1, 10, d_model) * 0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config['n_heads'],
            dim_feedforward=config['dim_feedforward'],
            dropout=config['dropout'],
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config['n_layers']
        )

        # Output layers
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               or (batch_size, seq_len, input_dim) for sequences

        Returns:
            Predicted scores of shape (batch_size, 1)
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = x.shape

        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Apply transformer
        x = self.transformer(x)

        # Pool over sequence (mean pooling)
        x = x.mean(dim=1)

        # Output projection
        x = self.output_norm(x)
        x = self.output_projection(x)

        return x