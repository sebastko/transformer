import torch
import math

def scaled_dot_product_attention(Q, K, V):
    batch, q_batch, d_q = Q.shape
    batch_, k_seq, d_k = K.shape
    batch__, v_seq, d_v = V.shape

    assert batch == batch_
    assert batch == batch__

    assert k_seq == v_seq
    assert d_q == d_k

    scaled_logits = torch.bmm(Q, K.transpose(1,2)) / torch.sqrt(torch.tensor(d_k))
    att = torch.softmax(scaled_logits, dim=-1)
    res = torch.bmm(att, V)
    assert res.shape == (batch, q_batch, d_v)
    return res


class SingleHeadAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, device):
        super().__init__()
        self.W_Q = torch.nn.Linear(d_model, d_k, bias=False, device=device, dtype=torch.float32)
        self.W_K = torch.nn.Linear(d_model, d_k, bias=False, device=device, dtype=torch.float32)
        self.W_V = torch.nn.Linear(d_model, d_v, bias=False, device=device, dtype=torch.float32)
    
    def forward(self, Q, K, V):
        return scaled_dot_product_attention(self.W_Q(Q), self.W_K(K), self.W_V(V))


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, d_k, d_v, num_heads, device):
        super().__init__()
        self.single_head_attentions = torch.nn.ModuleList([SingleHeadAttention(d_model, d_k, d_v, device) for _ in range(num_heads)])
        self.W_O = torch.nn.Linear(num_heads * d_v, d_model, bias=False, device=device, dtype=torch.float32)
    
    def forward(self, Q, K, V):
        heads = [att(Q, K, V) for att in self.single_head_attentions]
        concat_heads = torch.cat(heads, dim=-1)
        return self.W_O(concat_heads)


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, d_ff, num_heads, p_drop, device):
        super().__init__()
        assert d_model % num_heads == 0
        d_k = d_model // num_heads
        self.multi_head_attention = MultiHeadAttention(d_model, d_k, d_k, num_heads, device)
        self.dropout1 = torch.nn.Dropout(p_drop)
        self.norm1 = torch.nn.LayerNorm(d_model)

        self.ff1 = torch.nn.Linear(d_model, d_ff, bias=True, device=device, dtype=torch.float32)
        self.ff2 = torch.nn.Linear(d_ff, d_model, bias=True, device=device, dtype=torch.float32)
        self.dropout2 = torch.nn.Dropout(p_drop)
        self.norm2 = torch.nn.LayerNorm(d_model)
    
    def forward(self, x):
        y1 = self.multi_head_attention(x, x, x)
        y2 = self.norm1(x + self.dropout1(y1))

        y3 = self.ff2(torch.relu(self.ff1(y2)))
        y4 = self.norm2(y2 + self.dropout2(y3))
        return y4


class PositionalEmbeddings(torch.nn.Module):
    def __init__(self, num_emb, d_model, device):
        super().__init__()
        self._emb = torch.zeros((num_emb, d_model), dtype=torch.float32, device=device)
        for i in range(num_emb//2):
            for pos in range(d_model):
                self._emb[2*i,pos] = math.sin(pos/(10000**(2*i/d_model)))
        for j in range(0, num_emb, 2):
            i = j//2
            for pos in range(d_model):
                self._emb[2*i+1,pos] = math.cos(pos/(10000**(2*i/d_model)))
        self.device = device

    def forward(self, input):
        batch_size, seq_len = input.shape

        # omg, torch.range provides a range from start to stop INCLUSIVE :-o
        positions = torch.range(0, seq_len-1, dtype=torch.int)
        positions_batch = torch.stack([positions for _ in range(batch_size)]).to(self.device)
        return  torch.nn.functional.embedding(positions_batch, self._emb)


class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_input, max_len, n_layers, d_model, d_ff, num_heads, p_drop, device):
        super().__init__()
        self.token_emb = torch.nn.Embedding(d_input, d_model)
        self.pos_emb = PositionalEmbeddings(max_len, d_model, device)
        self.emb_dropout = torch.nn.Dropout(p_drop)
        ls = [TransformerEncoderLayer(d_model, d_ff, num_heads, p_drop, device) for _ in range(n_layers)]
        self.layers = torch.nn.Sequential(*ls)
        
    def forward(self, input):
        token_embs = self.token_emb(input)
        position_embs = self.pos_emb(input)
        embs = self.emb_dropout(token_embs + position_embs)
        outputs = self.layers(embs)
        return outputs
