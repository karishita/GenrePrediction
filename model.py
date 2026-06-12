import torch
import torch.nn as nn
from transformers import AutoModel

class TopicGenreModel(nn.Module):
    def __init__(self, num_topics, num_genres, vocab_size, model_name="roberta-base"):
        super().__init__()

        # 🔹 Backbone (RoBERTa)
        self.backbone = AutoModel.from_pretrained(model_name)
        bert_dim = self.backbone.config.hidden_size

        # 🔹 Context encoder ONLY (no BoW encoder)
        self.encoder = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 🔹 Latent space
        self.fc_mu = nn.Linear(256, num_topics)
        self.fc_logvar = nn.Linear(256, num_topics)

        # 🔹 Topic → word matrix
        self.beta = nn.Parameter(torch.randn(num_topics, vocab_size))
        nn.init.xavier_uniform_(self.beta)

        # 🔹 Topic → genre classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_topics, num_genres)
        )

    # ---------- Encode ----------
    def encode(self, embedding):
        h = self.encoder(embedding)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    # ---------- Reparameterization ----------
    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 10)  # stability
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---------- Decode ----------
    def decode(self, topic_dist):
       # beta_norm = torch.softmax(self.beta, dim=1)  
        logits = torch.matmul(topic_dist, self.beta)/0.5
        return torch.log_softmax(logits, dim=1)

    # ---------- Forward ----------
    def forward(self, input_ids, attention_mask):

        # 🔹 Get contextual embedding (MEAN pooling)
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        mask = attention_mask.unsqueeze(-1)
        embedding = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)

        # 🔹 Encode → latent topics
        mu, logvar = self.encode(embedding)

        # 🔹 Sample
        z = self.reparameterize(mu, logvar)
        #z = mu + 0.1 * torch.randn_like(mu)
        topic_dist = torch.softmax(z/0.5, dim=1)

        # 🔹 Genre prediction
        genre_logits = self.classifier(topic_dist)

        # 🔹 Word reconstruction
        word_log_probs = self.decode(topic_dist)

        return topic_dist, genre_logits, word_log_probs, mu, logvar