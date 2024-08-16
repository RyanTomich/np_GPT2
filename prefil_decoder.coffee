prompt: "Electric Light Orchestra"
tokenize: (3)
embed: (3, 768)
prefil
    decoder block 1
        ln_1
            (3, 768) x-mean/sqrt(var + epsilon)
            (3, 768) * (768,) + (768,)
            (3, 768)
        matrix_self_attn
            kqv = emb @ attn_weight + attn_bias:    (3, 2304) = (3, 768) @ (768, 2304) + (2304,)
            Qquery, key, value = split(kqv):        (3, 2304) = (3, 768), (3, 768), (3, 768)
            Q, K ,V = split_head(query, key, value)  = (12, 3, 64), (12, 3, 64), (12, 3, 64)
            w = Q @ K.t                              (12, 3, 3) = (12, 3, 64) @ (12, 64, 3)
            norm
            attn_mask
            softmax
            attn_output = attn_score @ V            (12, 3, 64) = (12, 3, 3) @ (12, 3, 64)
            merge_heads(attn_output)                (3, 768)
            context_proj = attn_output @ w + b      (3, 768) = (3, 768) @ (768, 768) + (768,)
        Residual
        ln_2
            (3, 768) = (3, 768) @ (768,) + (768,)
        MLP
            embl1 = emb @ weights + bias            (3, 3072) = (3, 768) @ (768, 3072) + (3072,)
            gelu
            embl2 = embl1 @ weights + bias         (3, 768) = (3, 3072) @ (3072, 768) + (768,)
        Residual

    decoder block 2
    decoder block 3
    ...

    norm        (3, 768) = (3, 768) * (768,) + (768,)
    logit       (3, 50257) = (3, 768) @ (50257, 768)

decoder
    decoder block 1
        ln_1
            (1, 768) = (1, 768) @ (768,) + (768,)
        matrix_self_attn
            kqv = emb @ attn_weight + attn_bias:    (1, 2304) = (1, 768) @ (768, 2304) + (2304,)
            query, key, value = split(kqv):         (3, 2304) = (1, 768), (1, 768), (1, 768)
            Q, K ,V = split_head(query, key, value) (12, 1, 64) (12, 4, 64) (12, 4, 64)
            w = Q @ K.t                             (12, 1, 4) = (12, 1, 64) @ (12, 64, 4)
            norm
            attn_mask
            softmax
            attn_output = attn_score @ V            (12, 1, 64) = (12, 1, 4) @ (12, 4, 64)
            merge_heads(attn_output)                (1, 768)
            context_proj = attn_output @ w + b      (1, 768) = (1, 768) @ (768, 768) + (768,)
        Residual
        ln_2
            (1, 768) = (1, 768) @ (768,) + (768,)
        MLP
            embl1 = emb @ weights + bias            (1, 3072) = (1, 768) @ (768, 3072) + (3072,)
            gelu
            embl2 = embl1 @ weights + bias          (1, 768) = (1, 3072) @ (3072, 768) + (768,)
        Residual

    decoder block 2
    decoder block 3
    ...

    norm        (1, 768) = (1, 768) * (768,) + (768,)
    logit       (1, 50257) = (1, 768) @ (50257, 768)
