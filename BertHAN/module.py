import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

BERT_dim = 768

class HAN(nn.Module):
    def __init__(self, flags):
        super(HAN, self).__init__()

        self.flags = flags
        self.embedding_dim = BERT_dim
        self.gru_dim = BERT_dim

        print("Loading FinBERT model...")
        self.bertmodel = AutoModel.from_pretrained(flags.modelpath)

        if flags.freeze:
            for param in self.bertmodel.parameters():
                param.requires_grad = False
            print("FinBERT parameters frozen")
        else:

            for param in self.bertmodel.parameters():
                param.requires_grad = False
            for param in self.bertmodel.encoder.layer[-1].parameters():
                param.requires_grad = True
            print("Only last BERT layer trainable")


        self.bi_gru = nn.GRU(
            self.embedding_dim,
            self.gru_dim,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(self.flags.dr)

        self.attn0 = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid(),
        )

        self.attn1 = nn.Sequential(
            nn.Linear(2 * self.gru_dim, 1),
            nn.Sigmoid(),
        )

        self.fc0 = nn.Linear(2 * self.gru_dim, self.flags.hidden_size)
        self.fc1 = nn.Linear(self.flags.hidden_size, self.flags.hidden_size)
        self.fc_out = nn.Linear(self.flags.hidden_size, self.flags.num_class)

        print(f"HAN model initialized - Hidden size: {flags.hidden_size}, Dropout: {flags.dr}")

    def forward(self, input):

        batch_size = input.shape[0]
        days = self.flags.days
        max_tweets = self.flags.max_num_tweets_len

        embeddings = torch.zeros(
            batch_size, days, max_tweets, self.embedding_dim,
            device=input.device
        )

        with torch.no_grad() if self.flags.freeze else torch.enable_grad():
            for day_idx in range(days):
                for tweet_idx in range(max_tweets):

                    outputs = self.bertmodel(
                        input_ids=input[:, day_idx, tweet_idx, 0, :],
                        token_type_ids=input[:, day_idx, tweet_idx, 1, :],
                        attention_mask=input[:, day_idx, tweet_idx, 2, :]
                    )

                    embeddings[:, day_idx, tweet_idx, :] = outputs.last_hidden_state[:, 0, :]

        attn_weights = self.attn0(embeddings)
        attn_weights = F.softmax(attn_weights, dim=2)

        day_embeddings = torch.sum(attn_weights * embeddings, dim=2)

        gru_output, _ = self.bi_gru(day_embeddings)

        temporal_attn_weights = self.attn1(gru_output)
        temporal_attn_weights = F.softmax(temporal_attn_weights, dim=1)

        final_representation = torch.sum(temporal_attn_weights * gru_output, dim=1)

        x = F.relu(self.fc0(final_representation))
        x = self.dropout(x) if self.training else x
        x = F.relu(self.fc1(x))
        x = self.dropout(x) if self.training else x
        output = self.fc_out(x)

        return output


class LightweightHAN(nn.Module):

    def __init__(self, flags):
        super(LightweightHAN, self).__init__()

        self.flags = flags
        self.embedding_dim = BERT_dim
        self.gru_dim = 256

        print("Loading FinBERT model (lightweight version)...")
        self.bertmodel = AutoModel.from_pretrained(flags.modelpath)

        for param in self.bertmodel.parameters():
            param.requires_grad = False

        self.gru = nn.GRU(
            self.embedding_dim,
            self.gru_dim,
            bidirectional=True,
            batch_first=True
        )

        self.dropout = nn.Dropout(self.flags.dr)

        self.attn0 = nn.Linear(self.embedding_dim, 1)
        self.attn1 = nn.Linear(self.gru_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.gru_dim, self.flags.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.flags.dr),
            nn.Linear(self.flags.hidden_size, self.flags.num_class)
        )

        print(f"Lightweight HAN initialized - GRU dim: {self.gru_dim}")

    def forward(self, input):
        batch_size = input.shape[0]
        days = self.flags.days
        max_tweets = self.flags.max_num_tweets_len

        embeddings = torch.zeros(
            batch_size, days, max_tweets, self.embedding_dim,
            device=input.device
        )

        with torch.no_grad():
            for day_idx in range(days):
                for tweet_idx in range(max_tweets):
                    outputs = self.bertmodel(
                        input_ids=input[:, day_idx, tweet_idx, 0, :],
                        token_type_ids=input[:, day_idx, tweet_idx, 1, :],
                        attention_mask=input[:, day_idx, tweet_idx, 2, :]
                    )
                    embeddings[:, day_idx, tweet_idx, :] = outputs.last_hidden_state[:, 0, :]

        attn_weights = torch.sigmoid(self.attn0(embeddings))
        attn_weights = F.softmax(attn_weights, dim=2)
        day_embeddings = torch.sum(attn_weights * embeddings, dim=2)


        gru_output, _ = self.gru(day_embeddings)

        temporal_weights = torch.sigmoid(self.attn1(gru_output))
        temporal_weights = F.softmax(temporal_weights, dim=1)
        final_repr = torch.sum(temporal_weights * gru_output, dim=1)

        output = self.classifier(final_repr)

        return output