"""Tiny character-level Transformer LLM trained on Python from this repo.

Core pieces only:
1. map characters to IDs
2. build next-character windows
3. train a small decoder Transformer (causal attention)
4. generate text from a prompt
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)

SEQ_LEN = 32
EMBED_DIM = 32
NUM_HEADS = 4
FF_DIM = 64
N_LAYERS = 2
DROPOUT = 0.1
EPOCHS = 12
BATCH_SIZE = 64
MAX_CHARS = 40_000
STRIDE = 2

REPO_ROOT = Path(__file__).resolve().parents[1]


class AddPosition(Layer):
    def __init__(self, seq_len, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.pos_emb = Embedding(seq_len, embed_dim)

    def call(self, x):
        positions = tf.range(tf.shape(x)[1])
        return x + self.pos_emb(positions)


class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads
        )
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout)
        self.drop2 = Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, use_causal_mask=True, training=training)
        x = self.norm1(x + self.drop1(attn_out, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.drop2(ffn_out, training=training))


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n").lower()


def load_repo_python(max_chars: int = MAX_CHARS) -> str:
    chunks = []
    seed = Path(__file__).with_name("simple_code_corpus.txt")
    if seed.exists():
        chunks.append(load_text(seed))

    skip = {Path(__file__).resolve()}
    for path in sorted(REPO_ROOT.rglob("*.py")):
        if path.resolve() in skip:
            continue
        chunks.append(load_text(path))
        if sum(len(part) for part in chunks) >= max_chars:
            break
    return "\n\n".join(chunks)[:max_chars]


def build_vocab(text: str):
    chars = sorted(set(text))
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for ch, i in char_to_id.items()}
    return chars, char_to_id, id_to_char


def make_sequences(text: str, char_to_id: dict, stride: int = STRIDE):
    ids = [char_to_id[ch] for ch in text]
    x, y = [], []
    limit = len(ids) - SEQ_LEN - 1
    for i in range(0, max(limit, 0), stride):
        x.append(ids[i : i + SEQ_LEN])
        y.append(ids[i + 1 : i + SEQ_LEN + 1])
    return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def build_model(vocab_size: int) -> Model:
    inputs = Input(shape=(SEQ_LEN,), dtype="int32")
    x = Embedding(vocab_size, EMBED_DIM)(inputs)
    x = AddPosition(SEQ_LEN, EMBED_DIM)(x)
    for _ in range(N_LAYERS):
        x = TransformerBlock(EMBED_DIM, NUM_HEADS, FF_DIM, DROPOUT)(x)
    outputs = Dense(vocab_size, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def generate(model, prompt: str, char_to_id: dict, id_to_char: dict, n_chars=160):
    seed = prompt.lower()
    unknown = [ch for ch in seed if ch not in char_to_id]
    if unknown:
        raise ValueError(f"Prompt has unseen characters: {unknown}")

    if len(seed) < SEQ_LEN:
        seed = seed.rjust(SEQ_LEN)
    else:
        seed = seed[-SEQ_LEN:]

    generated = seed
    for _ in range(n_chars):
        window = generated[-SEQ_LEN:]
        x = np.array([[char_to_id[ch] for ch in window]], dtype=np.int32)
        probs = model.predict(x, verbose=0)[0, -1]
        next_id = int(np.random.choice(len(probs), p=probs))
        generated += id_to_char[next_id]
    return generated.lstrip()


def main():
    text = load_repo_python()
    chars, char_to_id, id_to_char = build_vocab(text)
    x, y = make_sequences(text, char_to_id)

    print(f"corpus chars: {len(text)}")
    print(f"vocab size: {len(chars)}")
    print(f"training windows: {len(x)}")

    model = build_model(len(chars))
    model.summary()
    model.fit(x, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    prompt = "def "
    print("\n--- generated code ---")
    print(generate(model, prompt, char_to_id, id_to_char))


if __name__ == "__main__":
    tf.random.set_seed(42)
    np.random.seed(42)
    main()
