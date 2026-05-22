import torch

import torch.nn as nn
import torch.nn.functional as F

class MultiBranchPhishingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, cnn_filters, lstm_hidden, num_handcrafted_features, num_classes=1):
        super(MultiBranchPhishingModel, self).__init__()

        # Branch 1: URL Encoder (Embedding -> CNN -> BiLSTM)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)

        # CNN to capture local character/word-level patterns
        self.conv1d = nn.Conv1d(in_channels=embed_dim, out_channels=cnn_filters, kernel_size=3, padding=1)

        # BiLSTM to capture sequential dependencies
        self.bilstm = nn.LSTM(input_size=cnn_filters, hidden_size=lstm_hidden,
                              batch_first=True, bidirectional=True)

        # Branch 2: Feature Encoder for handcrafted features
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_handcrafted_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Fusion & Classification Layer
        # bilstm output is hidden_size * 2 (bidirectional)
        fusion_dim = (lstm_hidden * 2) + 32

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
            # nn.Sigmoid() # Typically handled by BCEWithLogitsLoss during training
        )

    def forward(self, url_seq, handcrafted_features):
        # Branch 1
        # url_seq: (batch_size, seq_len)
        x_emb = self.embedding(url_seq) # (batch_size, seq_len, embed_dim)

        # Conv1d expects (batch_size, channels, seq_len)
        x_emb = x_emb.permute(0, 2, 1)
        x_conv = F.relu(self.conv1d(x_emb)) # (batch_size, cnn_filters, seq_len)

        # LSTM expects (batch_size, seq_len, input_size)
        x_conv = x_conv.permute(0, 2, 1)
        lstm_out, (hn, cn) = self.bilstm(x_conv) # lstm_out: (batch_size, seq_len, lstm_hidden*2)

        # We can take the last hidden state or pool the outputs. Let's take the last output.
        # Alternatively, max pooling over sequence
        url_rep, _ = torch.max(lstm_out, dim=1) # (batch_size, lstm_hidden*2)

        # Branch 2
        feat_rep = self.feature_encoder(handcrafted_features) # (batch_size, 32)

        # Fusion
        fused = torch.cat((url_rep, feat_rep), dim=1) # (batch_size, lstm_hidden*2 + 32)

        # Classification
        out = self.classifier(fused) # (batch_size, 1)

        return out

if __name__ == "__main__":
    # Test the model with dummy data
    vocab_size = 100
    embed_dim = 32
    cnn_filters = 64
    lstm_hidden = 64
    num_hc_features = 15
    batch_size = 8
    seq_len = 50

    model = MultiBranchPhishingModel(vocab_size, embed_dim, cnn_filters, lstm_hidden, num_hc_features)

    dummy_seq = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_feats = torch.randn(batch_size, num_hc_features)

    output = model(dummy_seq, dummy_feats)
    print(f"Output shape: {output.shape}") # Should be [8, 1]
