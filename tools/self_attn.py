import numpy as np
from .linear import Linear, SoftMax
from .norm import LayerNorm
from .activation import ReLU # , Sigmoid, Tanh
from .dropout import Dropout
from .embedding import Embedding

class SelfAttention:
    def __init__(self, input_dim, heads):
        # super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.heads = heads
        self.head_dim = input_dim // heads
        
        self.softmax = SoftMax(axis=3)

        self.values = Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = Linear(heads * self.head_dim, input_dim, bias=False)

    def forward(self, value, key, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = value.shape[1], key.shape[1], query.shape[1]

        values = value.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # queries_transpose = queries.transpose(0, 2, 1, 3) , (N, heads, query_len, head_dim)
        # keys_transpose = keys.transpose(0, 2, 3, 1)       , (N, heads, head_dim, key_len)
        # energy = queries_transpose @ keys_transpose       , (N, heads, query_len, key_len)
        energy = np.einsum("nqhd,nkhd->nhqk", queries, keys)

        if mask is not None:
            # energy = energy.masked_fill(mask == 0, float("-1e20"))
            energy = np.where(mask == 0, -1e20, energy)

        # Attenthon(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V 
        # attention                                         , (N, heads, query_len, key_len)
        # values                                            , (N, value_len, heads, head_dim)
        # out = attention @ values                          , (N, query_len, heads, head_dim)
        attention = self.softmax(energy / (self.input_dim ** (1 / 2)))
        out = np.einsum("nhql,nlhd->nqhd", attention, values)
        # concat
        out = out.reshape(N, query_len, self.heads * self.head_dim)

        out = self.fc_out(out)
        return out
    
    def __call__(self, value, key, query, mask):
        return self.forward(value, key, query, mask)
    
class TransformerBlock:
    def __init__(self, input_dim, heads, dropout, forward_expansion):
        # super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(input_dim, heads)
        self.norm1 = LayerNorm(input_dim)
        self.norm2 = LayerNorm(input_dim)

        # ffn : linear - relu - linear
        self.feed_forward1 = Linear(input_dim, forward_expansion*input_dim)
        self.feed_forward2 = ReLU()
        self.feed_forward3 = Linear(forward_expansion*input_dim, input_dim)

        self.dropout = Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # residual connection
        # dropout(attention + query)
        x = self.norm1(attention + query)
        x = self.dropout(x)

        # feed forward
        forward = self.feed_forward1(x)
        forward = self.feed_forward2(forward)
        out = self.feed_forward3(forward)

        # residual connection
        # dropout(out + x)
        out = self.norm2(out + x)
        out = self.dropout(out)

        return out
    
    def __call__(self, value, key, query, mask):
        return self.forward(value, key, query, mask)
    
class Encoder:
    def __init__(
            self,
            src_vocab_size,
            input_dim,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        # super(Encoder, self).__init__()
        self.embed_dim = input_dim
        self.device = device
        self.word_embedding = Embedding(src_vocab_size, input_dim)
        self.position_embedding = Embedding(max_length, input_dim)

        self.layers = []
        for i in range(num_layers):
            layer = TransformerBlock(
                    input_dim=input_dim, 
                    heads=heads, 
                    dropout=dropout, 
                    forward_expansion=forward_expansion
                )
            self.layers.append(layer)
            setattr(self, f"encoder_layer_{i}", layer)
        
        self.dropout = Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = np.tile(np.arange(0, seq_length), (N, 1)) # np.arange(0, seq_length).expand(N, seq_length)

        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            out = layer.forward(x, x, x, mask)

        return out
    
    def __call__(self, x, mask):
        return self.forward(x, mask)
    
class DecoderBlock:
    def __init__(self, input_dim, heads, forward_expansion, dropout):
        # super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(input_dim, heads)
        self.norm = LayerNorm(input_dim)
        self.transformer_block = TransformerBlock(
            input_dim=input_dim, 
            heads=heads, 
            dropout=dropout, 
            forward_expansion=forward_expansion
        )
        self.dropout = Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.norm(attention + x)
        query = self.dropout(query)

        out = self.transformer_block(value, key, query, src_mask)

        return out
    
    def __call__(self, x, value, key, src_mask, trg_mask):
        return self.forward(x, value, key, src_mask, trg_mask)
    
class Decoder:
    def __init__(
            self,
            trg_vocab_size,
            input_dim,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device
    ):
        # super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = Embedding(trg_vocab_size, input_dim)
        self.position_embedding = Embedding(max_length, input_dim)

        self.layers = []
        for i in range(num_layers):
            layer = DecoderBlock(
                        input_dim=input_dim, 
                        heads=heads, 
                        forward_expansion=forward_expansion, 
                        dropout=dropout
                    )
            self.layers.append(layer)
            setattr(self, f"decoder_layer_{i}", layer)

        self.fc_out = Linear(input_dim, trg_vocab_size)
        self.dropout = Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = np.tile(np.arange(0, seq_length), (N, 1)) # positions = np.arange(0, seq_length).expand_dims(N, seq_length).to(self.device)

        x = self.word_embedding(x) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out
    
    def __call__(self, x, enc_out, src_mask, trg_mask):
        return self.forward(x, enc_out, src_mask, trg_mask)
    
class Transformer:
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            input_dim = 256,
            num_layers = 6,
            forward_expansion = 4,
            heads = 8,
            dropout = 0,
            max_length = 100,
            device = "mps"
    ):
        # super(Transformer, self).__init__()
        self.encoder = Encoder(
                src_vocab_size=src_vocab_size, 
                input_dim=input_dim, 
                num_layers=num_layers, 
                heads=heads, 
                device=device, 
                forward_expansion=forward_expansion, 
                dropout=dropout, 
                max_length=max_length
            )
        
        self.decoder = Decoder(
                trg_vocab_size=trg_vocab_size, 
                input_dim=input_dim, 
                num_layers=num_layers, 
                heads=heads, 
                forward_expansion=forward_expansion, 
                dropout=dropout, 
                max_length=max_length, 
                device=device
            )
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        # (N, 1, 1, src_len)
        src_mask = (src != self.src_pad_idx)[:, np.newaxis, np.newaxis, :]
        return src_mask
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = np.tril(np.ones((trg_len, trg_len)))
        trg_mask = np.broadcast_to(trg_mask, (N, 1, trg_len, trg_len))
        return trg_mask
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)

        return out
    
    def __call__(self, src, trg):
        return self.forward(src, trg)