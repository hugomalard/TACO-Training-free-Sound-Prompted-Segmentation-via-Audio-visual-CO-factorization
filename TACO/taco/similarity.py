"""Cosine similarities used by the semantic penalty and class readout."""

import torch


def get_closest(image_features, text_features):
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features @ image_features.T


def compute_clap_similarity(audio_embeddings, text_embeddings, logit_scale):
    audio_embeddings = audio_embeddings / torch.norm(audio_embeddings, dim=-1, keepdim=True)
    text_embeddings = text_embeddings / torch.norm(text_embeddings, dim=-1, keepdim=True)
    similarity = logit_scale * text_embeddings @ audio_embeddings.T
    return similarity.T
