"""Audio-visual co-factorization.

Sigmoid activations, a cross-entropy penalty on the closest factor, and an
optional temporal cosine penalty between consecutive frames.
"""

import torch
from torch_kmeans import KMeans

from taco.similarity import compute_clap_similarity, get_closest


def _init_factors(tokens, k):
    kmeans = KMeans(n_clusters=k, verbose=False, init_method="k-means++")
    centers = kmeans(tokens).centers.detach().to(tokens.device)
    centers.requires_grad_(True)
    weights = torch.randn(
        (tokens.shape[0], tokens.shape[1], k),
        requires_grad=True,
        device=tokens.device,
    )
    return weights, centers


def _semantic_components(tokens, activations, k):
    # activations: (B, tokens, K) -> weighted token means (B, K, dim)
    weights = torch.swapaxes(activations, 1, 2).unsqueeze(3)
    expanded = tokens.unsqueeze(1).repeat(1, k, 1, 1) * weights
    return expanded.mean(dim=2)


def _closest_factor_loss(image_sim, audio_sim, batch, k):
    criterion = torch.nn.CrossEntropyLoss(reduction="none")
    image_ce = criterion(image_sim, audio_sim.softmax(dim=1)).view(batch, k)
    audio_ce = criterion(audio_sim, image_sim.softmax(dim=1)).view(batch, k)
    image_penalty = image_ce.topk(2, dim=1, largest=False).values[:, 0].squeeze()
    audio_penalty = audio_ce.topk(2, dim=1, largest=False).values[:, 0].squeeze()
    return image_ce, image_penalty, audio_penalty


def _temporal_penalty(image_factors, audio_factors, image_ce):
    if image_factors.shape[0] < 2:
        return image_factors.new_zeros(())
    chosen = image_ce.argmin(dim=1).view(-1)
    penalty = image_factors.new_zeros(())
    for index in range(image_factors.shape[0] - 1):
        current = chosen[index]
        nxt = chosen[index + 1]
        penalty = penalty - torch.nn.functional.cosine_similarity(
            image_factors[index, current], image_factors[index + 1, nxt], dim=0
        )
        penalty = penalty - torch.nn.functional.cosine_similarity(
            audio_factors[index, current], audio_factors[index + 1, nxt], dim=0
        )
    return penalty


def _gd_step(parameters, lr, positivity):
    with torch.no_grad():
        for param in parameters:
            param -= lr * param.grad
        if positivity:
            parameters[2].clamp_(min=0)
            parameters[3].clamp_(min=0)
        for param in parameters:
            param.grad.zero_()


def conmf_txt(
    image_tokens,
    audio_tokens,
    k,
    b1=1,
    b2=1,
    b3=1,
    b4=1,
    max_iter=200,
    lr=0.01,
    clap_text=None,
    clip_text=None,
    clap_model=None,
    positivity=False,
    temporal=False,
):
    """Co-factorize image and audio tokens against a shared text bank.

    Audio components are mapped with the CLAP projection before the penalty.
    Returns sigmoid activations, factors, and the chosen factor index.
    """
    batch = image_tokens.shape[0]
    image_w, image_h = _init_factors(image_tokens, k)
    audio_w, audio_h = _init_factors(audio_tokens, k)
    device = image_tokens.device
    text_audio = clap_text.detach().to(device)
    text_image = clip_text.detach().to(device)
    logit_scale = clap_model.module.clap.logit_scale.detach().exp().to(device)
    project = clap_model.module.clap.audio_encoder.projection
    image_act = audio_act = image_ce = None

    for step in range(max_iter):
        if step % 500 == 0 and step != 0:
            lr = lr / 2

        image_act = torch.sigmoid(image_w)
        audio_act = torch.sigmoid(audio_w)
        recon_image = torch.norm(
            image_tokens.detach() - torch.bmm(image_act, image_h), "fro", dim=(1, 2)
        )
        recon_audio = torch.norm(
            audio_tokens.detach() - torch.bmm(audio_act, audio_h), "fro", dim=(1, 2)
        )

        audio_component = _semantic_components(audio_tokens, audio_act, k)
        image_component = _semantic_components(image_tokens.detach(), image_act, k)
        audio_sim = compute_clap_similarity(
            project(audio_component.flatten(0, 1)),
            text_audio,
            logit_scale,
        )
        image_sim = get_closest(image_component.flatten(0, 1), text_image).T
        image_ce, image_penalty, audio_penalty = _closest_factor_loss(
            image_sim, audio_sim, batch, k
        )

        temporal_penalty = image_tokens.new_zeros(())
        if temporal:
            temporal_penalty = _temporal_penalty(image_h, audio_h, image_ce)

        error = (
            recon_image * b1
            + recon_audio * b2
            + image_penalty * b3
            + audio_penalty * b4
            + temporal_penalty
        )
        error.sum().backward()
        _gd_step((image_w, audio_w, image_h, audio_h), lr, positivity)

    # Last forward pass, before the gradient step.
    chosen = image_ce.argmin(dim=1)
    return image_act, image_h, audio_act, audio_h, chosen, image_ce


def conmf_bary(
    image_tokens,
    audio_tokens,
    k,
    b1=1,
    b2=1,
    b3=1,
    b4=1,
    b5=1,
    max_iter=200,
    lr=0.01,
    barycenters=None,
    positivity=False,
    clap_model=None,
):
    """Co-factorize against precomputed CLAP/CLIP barycenters.

    Audio components are compared in token space, without the CLAP projection.
    ``b5`` is unused.
    """
    del b5
    batch = image_tokens.shape[0]
    image_w, image_h = _init_factors(image_tokens, k)
    audio_w, audio_h = _init_factors(audio_tokens, k)
    device = image_tokens.device
    barycenters = barycenters.detach().to(device)
    text_audio = barycenters[:, 0:768]
    text_image = barycenters[:, 768:]
    logit_scale = clap_model.module.clap.logit_scale.detach().exp().to(device)
    image_act = audio_act = image_ce = None

    for step in range(max_iter):
        if step % 500 == 0 and step != 0:
            lr = lr / 2

        image_act = torch.sigmoid(image_w)
        audio_act = torch.sigmoid(audio_w)
        recon_image = torch.norm(
            image_tokens.detach() - torch.bmm(image_act, image_h), "fro", dim=(1, 2)
        )
        recon_audio = torch.norm(
            audio_tokens.detach() - torch.bmm(audio_act, audio_h), "fro", dim=(1, 2)
        )
        audio_component = _semantic_components(audio_tokens, audio_act, k)
        image_component = _semantic_components(image_tokens.detach(), image_act, k)
        audio_sim = compute_clap_similarity(
            audio_component.flatten(0, 1), text_audio, logit_scale
        )
        image_sim = get_closest(image_component.flatten(0, 1), text_image).T
        # Computed and discarded.
        get_closest(image_h.flatten(0, 1), text_image)
        compute_clap_similarity(audio_h.flatten(0, 1), text_audio, logit_scale)
        image_ce, image_penalty, audio_penalty = _closest_factor_loss(
            image_sim, audio_sim, batch, k
        )
        error = recon_image * b1 + recon_audio * b2 + image_penalty * b3 + audio_penalty * b4
        error.sum().backward()
        _gd_step((image_w, audio_w, image_h, audio_h), lr, positivity)

    chosen = image_ce.argmin(dim=1)
    return image_act, image_h, audio_act, audio_h, chosen, image_ce
