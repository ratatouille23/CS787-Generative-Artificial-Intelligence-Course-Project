import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import Word2Vec
import pickle
import os


class OriginalHAN(nn.Module):

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=50,
                 num_classes=2, dropout=0.3, pretrained_embeddings=None):
        super(OriginalHAN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings

        self.word_gru = nn.GRU(embedding_dim, hidden_dim,
                               bidirectional=True, batch_first=True)

        self.word_attention = nn.Linear(2 * hidden_dim, 1)

        self.sentence_gru = nn.GRU(2 * hidden_dim, hidden_dim,
                                   bidirectional=True, batch_first=True)

        self.sentence_attention = nn.Linear(2 * hidden_dim, 1)

        self.temporal_gru = nn.GRU(2 * hidden_dim, hidden_dim,
                                   bidirectional=True, batch_first=True)

        self.temporal_attention = nn.Linear(2 * hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        print(f"Original HAN initialized:")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Dropout: {dropout}")

    def forward(self, x):
        batch_size, num_days, num_sentences, num_words = x.shape

        x = x.view(batch_size * num_days * num_sentences, num_words)

        embedded = self.embedding(x)

        word_output, _ = self.word_gru(embedded)

        word_attn_weights = torch.tanh(self.word_attention(word_output))
        word_attn_weights = F.softmax(word_attn_weights, dim=1)
        sentence_repr = torch.sum(word_attn_weights * word_output, dim=1)

        sentence_repr = sentence_repr.view(batch_size * num_days, num_sentences, -1)

        sentence_output, _ = self.sentence_gru(sentence_repr)

        sent_attn_weights = torch.tanh(self.sentence_attention(sentence_output))
        sent_attn_weights = F.softmax(sent_attn_weights, dim=1)
        day_repr = torch.sum(sent_attn_weights * sentence_output, dim=1)

        day_repr = day_repr.view(batch_size, num_days, -1)

        temporal_output, _ = self.temporal_gru(day_repr)

        temp_attn_weights = torch.tanh(self.temporal_attention(temporal_output))
        temp_attn_weights = F.softmax(temp_attn_weights, dim=1)
        final_repr = torch.sum(temp_attn_weights * temporal_output, dim=1)

        x = F.relu(self.fc1(final_repr))
        x = self.dropout(x) if self.training else x
        output = self.fc2(x)

        return output


class LightweightOriginalHAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=50, hidden_dim=32,
                 num_classes=2, dropout=0.3, pretrained_embeddings=None):
        super(LightweightOriginalHAN, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))

        self.word_gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.sentence_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.word_attention = nn.Linear(hidden_dim, 1)
        self.sentence_attention = nn.Linear(hidden_dim, 1)
        self.temporal_attention = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        print(f"Lightweight Original HAN initialized:")
        print(f"  Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}")

    def forward(self, x):
        batch_size, num_days, num_sentences, num_words = x.shape

        x = x.view(batch_size * num_days * num_sentences, num_words)
        embedded = self.embedding(x)
        word_output, _ = self.word_gru(embedded)

        word_attn = torch.sigmoid(self.word_attention(word_output))
        word_attn = F.softmax(word_attn, dim=1)
        sentence_repr = torch.sum(word_attn * word_output, dim=1)

        sentence_repr = sentence_repr.view(batch_size * num_days, num_sentences, -1)
        sentence_output, _ = self.sentence_gru(sentence_repr)

        sent_attn = torch.sigmoid(self.sentence_attention(sentence_output))
        sent_attn = F.softmax(sent_attn, dim=1)
        day_repr = torch.sum(sent_attn * sentence_output, dim=1)

        day_repr = day_repr.view(batch_size, num_days, -1)
        temporal_output, _ = self.temporal_gru(day_repr)

        temp_attn = torch.sigmoid(self.temporal_attention(temporal_output))
        temp_attn = F.softmax(temp_attn, dim=1)
        final_repr = torch.sum(temp_attn * temporal_output, dim=1)

        output = self.classifier(final_repr)
        return output


def build_word2vec_embeddings(texts, embedding_dim=100, min_count=2, window=5):
    print("Training Word2Vec model...")

    model = Word2Vec(texts, vector_size=embedding_dim, window=window,
                     min_count=min_count, workers=4, sg=1)  # sg=1 for skip-gram

    vocab = model.wv.index_to_key
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}  # Reserve 0 for padding
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = len(word2idx)

    vocab_size = len(word2idx)
    embeddings = np.zeros((vocab_size, embedding_dim))

    for word, idx in word2idx.items():
        if word in model.wv:
            embeddings[idx] = model.wv[word]
        else:
            embeddings[idx] = np.random.randn(embedding_dim) * 0.01

    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dim: {embedding_dim}")

    return embeddings, word2idx


def save_vocabulary(word2idx, filepath='vocab.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump(word2idx, f)
    print(f"Vocabulary saved to {filepath}")


def load_vocabulary(filepath='vocab.pkl'):
    with open(filepath, 'rb') as f:
        word2idx = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return word2idx