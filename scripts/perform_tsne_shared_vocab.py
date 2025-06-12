## perform tsne beforehand for each wave

import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


def intersection_align_gensim(m1, m2):
    common = list(set(m1.wv.key_to_index).intersection(set(m2.wv.key_to_index)))
    common.sort(
        key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"),
        reverse=True,
    )

    def align_model(model):
        vectors = np.array([model.wv[w] for w in common])
        model.wv.vectors = vectors
        model.wv.key_to_index = {w: i for i, w in enumerate(common)}
        model.wv.index_to_key = common
        return model

    return align_model(m1), align_model(m2)


def smart_procrustes_align_gensim(base, other):
    base, other = intersection_align_gensim(base, other)
    m = other.wv.vectors.T @ base.wv.vectors
    u, _, v = np.linalg.svd(m)
    ortho = u @ v
    other.wv.vectors = other.wv.vectors @ ortho
    return other


def cosine_distance(a, b):
    return (1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) / 2


if __name__ == "__main__":
    wave_1 = Word2Vec.load(
        "../5ysplits_models/diachronic_wave_1_processed_model.bin"
    )
    wave_2 = Word2Vec.load(
        "../5ysplits_models/diachronic_wave_2_processed_model.bin"
    )
    wave_3 = Word2Vec.load(
        "../5ysplits_models/diachronic_wave_3_processed_model.bin"
    )
    wave_4 = Word2Vec.load(
        "../5ysplits_models/diachronic_wave_4_processed_model.bin"
    )
    wave_5 = Word2Vec.load(
        "../5ysplits_models/diachronic_wave_5_processed_model.bin"
    )

    waves = {0: wave_1, 1: wave_2, 2: wave_3, 3: wave_4, 4: wave_5}

    models = [waves[key] for key in sorted(waves.keys())]

    periods = sorted(waves.keys())

    for i in reversed(range(1, len(models))):
        models[i - 1] = smart_procrustes_align_gensim(models[i], models[i - 1])

    waves_tsne = {}

    for idx, period in enumerate(periods):
        print(f"Processing wave {period + 1}...")
        tsne = TSNE(n_components=2, random_state=42, max_iter=1000)
        word_vectors = waves[period].wv.vectors
        waves_tsne[idx] = tsne.fit_transform(word_vectors)

    # Save the t-SNE embeddings for each wave
    for idx, tsne_embedding in waves_tsne.items():
        np.save(
            f"/home/ddukic/retriever-diachronic-emb/data/processed/tsne_embeddings_wave_{idx + 1}.npy",
            tsne_embedding,
        )
