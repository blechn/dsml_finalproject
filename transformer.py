import jax
import jax.numpy as jnp

import flax
import flax.linen as nn


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 5000
    wavelength: float = 10000.0

    def setup(self):
        pe = jnp.zeros((self.max_len, self.d_model))
        pos = jnp.arange(0, self.max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2)
            * (-(jnp.log(self.wavelength) / self.d_model))
        )

        pe = pe.at[:, 0::2].set(jnp.sin(pos * div_term))
        pe = pe.at[:, 1::2].set(
            jnp.cos(pos * div_term)
            if ((self.d_model % 2) == 0)
            else jnp.cos(pos * div_term)[:, :-1]
        )
        self.pe = pe[jnp.newaxis, ...]

    def __call__(self, x):
        return x + self.pe[:, : x.shape[-2]]


class EncoderBlock(nn.Module):
    d_model: int
    n_heads: int
    ff_dim: int = 128
    use_mask: bool = False
    dropout_rate: float = 0.2

    def setup(self):
        self.mhattention = nn.MultiHeadAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )
        self.linear = [
            nn.Dense(self.ff_dim, kernel_init=nn.initializers.xavier_uniform()),
            nn.Dropout(self.dropout_rate),
            nn.relu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform()),
        ]

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, train=True):
        seq_len = x.shape[-2]
        mask = (
            jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
            if self.use_mask
            else None
        )
        attn_out = self.mhattention(x, x, x, mask=mask, deterministic=not train)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm1(x)

        lin_out = x
        for l in self.linear:
            lin_out = (
                l(lin_out)
                if not isinstance(l, nn.Dropout)
                else l(lin_out, deterministic=not train)
            )
        x = x + self.dropout(lin_out, deterministic=not train)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    d_model: int
    n_layers: int
    n_heads: int
    ff_dim: int = 128
    use_mask: bool = False
    dropout_rate: float = 0.2

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                use_mask=self.use_mask,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x, train=True):
        for l in self.layers:
            x = l(x, train=train)
        return x


class DecoderBlock(nn.Module):
    d_model: int
    n_heads: int
    ff_dim: int
    use_mask: bool = True
    dropout_rate: float = 0.2

    def setup(self):
        self.mhattention = nn.MultiHeadAttention(
            self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )
        self.mhattention2 = nn.MultiHeadAttention(
            self.n_heads,
            qkv_features=self.d_model,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )
        self.linear = [
            nn.Dense(self.ff_dim, kernel_init=nn.initializers.xavier_uniform()),
            nn.Dropout(self.dropout_rate),
            nn.relu,
            nn.Dense(self.d_model, kernel_init=nn.initializers.xavier_uniform()),
        ]

        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.norm3 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x: jnp.ndarray, y: jnp.ndarray, train: bool = True):
        assert x.shape[-2] == y.shape[-2], "Shapes don't match wtf"
        seq_len = y.shape[-2]
        mask = (
            jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
            if self.use_mask
            else None
        )
        attn_out = self.mhattention(y, y, y, mask=mask, deterministic=not train)
        y = y + self.dropout(attn_out, deterministic=not train)
        y = self.norm1(y)

        attn_out = self.mhattention2(x, x, y, mask=None, deterministic=not train)
        x = x + self.dropout(attn_out, deterministic=not train)
        x = self.norm2(x)

        lin_out = x
        for l in self.linear:
            lin_out = (
                l(lin_out)
                if not isinstance(l, nn.Dropout)
                else l(lin_out, deterministic=not train)
            )
        x = x + self.dropout(lin_out, deterministic=not train)
        x = self.norm3(x)
        return x


class TransformerDecoder(nn.Module):
    d_model: int
    n_layers: int
    n_heads: int
    ff_dim: int = 128
    use_mask: bool = True
    dropout_rate: float = 0.2

    def setup(self):
        self.layers = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ff_dim=self.ff_dim,
                use_mask=self.use_mask,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, src, tgt, train: bool = True):
        for l in self.layers:
            tgt = l(src, tgt, train=train)
        return tgt


class Transformer(nn.Module):
    d_model: int
    n_heads: int
    n_encoders: int
    n_decoders: int
    ff_dim: int = 128
    use_encoder_mask: bool = False
    use_decoder_mask: bool = True
    pe_max_len: int = 5000
    dropout_rate: float = 0.2

    def setup(self):
        self.pos_enc = PositionalEncoding(d_model=self.d_model, max_len=self.pe_max_len)
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            n_layers=self.n_encoders,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            use_mask=self.use_encoder_mask,
            dropout_rate=self.dropout_rate,
        )
        self.decoder = TransformerDecoder(
            d_model=self.d_model,
            n_layers=self.n_decoders,
            n_heads=self.n_heads,
            ff_dim=self.ff_dim,
            use_mask=self.use_decoder_mask,
            dropout_rate=self.dropout_rate,
        )
        self.linear = nn.Dense(self.d_model)

    def __call__(self, x, y, train: bool = True):
        x = self.pos_enc(x)
        y = self.pos_enc(y)
        x = self.encoder(x, train=train)
        x = self.decoder(x, y, train=train)
        out = self.linear(x)
        return out


def get_model(
    model_str: str | None = None,
    n_encoders: int = 2,
    n_decoders: int = 2,
    ff_dim: int = 128,
    use_encoder_mask: bool = False,
    use_decoder_mask: bool = True,
    pe_max_len: int = 1000,
    dropout_rate: float = 0.2,
):
    match model_str:
        case None:  # Default: return a small model for lorenz63
            return Transformer(3, 3, 2, 2, ff_dim=64, pe_max_len=100)
        case "lorenz63":
            return Transformer(
                3,
                3,
                n_encoders=n_encoders,
                n_decoders=n_decoders,
                ff_dim=ff_dim,
                use_encoder_mask=use_encoder_mask,
                use_decoder_mask=use_decoder_mask,
                pe_max_len=pe_max_len,
                dropout_rate=dropout_rate,
            )
        case "lorenz96":
            return Transformer(
                20,
                10,
                n_encoders=n_encoders,
                n_decoders=n_decoders,
                ff_dim=ff_dim,
                use_encoder_mask=use_encoder_mask,
                use_decoder_mask=use_decoder_mask,
                pe_max_len=pe_max_len,
                dropout_rate=dropout_rate,
            )
        case _:  # Default: return a small model for lorenz63
            return Transformer(3, 3, 2, 2, ff_dim=64, pe_max_len=100)