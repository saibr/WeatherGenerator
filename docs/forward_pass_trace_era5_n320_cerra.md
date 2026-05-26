# Complete Forward-Pass Trace: `era5_n320_cerra` Stream

> Generated from source analysis of `src/weathergen/`, `config/config_forecasting.yml`,  
> `config/streams/era5_n320_cerra/era5.yml`, and `config/streams/era5_n320_cerra/cerra.yml`.

---

## Notation and Global Constants

| Symbol | Value | Source |
|--------|-------|--------|
| `C` | 12,288 | `12 * 4^5` (healpix_level=5) |
| `D_local` | 2,048 | `ae_local_dim_embed` |
| `D_global` | 2,048 | `ae_global_dim_embed` |
| `Q` | 1 | `ae_local_num_queries` |
| `T_e` | 64 | ERA5 `token_size` |
| `T_c` | 512 | CERRA `token_size` |
| `D_stream` | 512 | per-stream `embed.dim_embed` |
| `D_coord` | 512 | `embed_target_coords.dim_embed` |
| `P_e` | ≈542,080 | ERA5 N320 reduced Gaussian grid points |
| `P_c` | ≈500,000 | CERRA 5.5km European domain grid points |
| `B` | batch size | (1 per GPU in FSDP; here B=1 for clarity) |
| `n_steps` | 1 | input time steps (one 6h window) |
| `rs` | `B × n_steps` = B | collapsed batch×step dimension |
| `dtype_compute` | `bfloat16` | `attention_dtype`, `mixed_precision_dtype` |
| `dtype_params` | `float32` | weights stored in float32 |

---

## Stage 0 — Raw Data Ingestion

### Source: `DataReaderAnemoi`

```
ERA5  raw tensor: [P_e × n_steps, F_e_raw]   float32   CPU
CERRA raw tensor: [P_c × n_steps, F_c_raw]   float32   CPU
```

**Layout:** `data.transpose([0,2,1]).reshape(...)` from zarr → coords-first; shape is `[n_steps * N_grid, N_channels]`.

**Normalization (input):**  
Applied at read time via anemoi dataset statistics:
```
data_normalized = (data - mean[channel]) / stdev[channel]
```
- `mean` and `stdev` from `ds.statistics` in the zarr — these are **global, per-variable, per-level statistics** computed over the entire training period (1979–2022 for ERA5).
- No seasonal anomaly, no climatology subtraction. Raw anomalies relative to training-set mean.
- `geoinfo` channels (orog, lsm) are separately normalized using `mean_geoinfo` / `stdev_geoinfo`.
- **NaN handling:** Land-sea masking creates structural NaNs in some ERA5 ocean/land fields. NaNs in `source_tokens_cells` are zeroed (`mask_value = 0.0`) before embedding — they are therefore **not masked out of attention** but replaced with the zero embedding value.

**Channel selection (`select_channels`):**

| Stream | Excluded | Approx. remaining source channels `F_s` |
|--------|----------|----------------------------------------|
| ERA5   | `w_*`, `skt`, `tcw`, `cp`, `tp` | ≈163 (pressure-level u,v,t,q,z + surface fields) |
| CERRA  | `skt`, `tciwv`, `tp`, `al`, `rsn`, `sde`, `sf` | **85** (pressure-level t,u,v,r,z + surface/radiation fields) |

**Geoinfo channels:** `['orog', 'lsm']` → 2 channels per stream.

**Gradient:** None. Raw data ingestion is CPU-only preprocessing.

**Memory:** ERA5 raw `[542080, ~163]` float32 ≈ **354 MB**; CERRA raw `[500000, 85]` float32 ≈ **170 MB**.

---

## Stage 1 — Tokenization

### Entry point: `tokenize_spacetime` → `hpy_splits` (in `tokenizer_utils.py`)

```
Input:  ReaderData(coords=[P_s × n_steps, 2], geoinfos=[P_s × n_steps, 2],
                   data=[P_s × n_steps, F_s], datetimes=[P_s × n_steps])

Output: idxs_cells    : list[list[int]]  shape [C][n_tokens_per_cell][token_size]
        idxs_cells_lens: list[list[int]] shape [C][n_tokens_per_cell]
```

**Algorithm:**
1. Convert (lat, lon) → HEALPix NESTED pixel index at level 5 using `ang2pix`.
2. Sort points by pixel index; `argsort(stable=True)` within each cell by latitude.
3. Split points into chunks of exactly `token_size` (padding last chunk with index 0 = a zero-vector pad).

**ERA5 tokenization characteristics:**

```
avg points per HEALPix cell:  542,080 / 12,288 ≈ 44
token_size = 64
→ 44 < 64, so exactly 1 token per cell per time step
→ last ~20 slots in that token are zero-padded
N_tok_ERA5 ≈ C × 1 = 12,288  (before masking)
```

With `tokenize_spacetime=True` and 6-hourly ERA5, each window has 1 time step → same count.

**CERRA tokenization characteristics:**

```
CERRA covers ~Europe; only ~1,600–2,500 HEALPix cells have data
avg points per populated cell:  ≈250–1700 (varies strongly by cell)
token_size = 512
cells with < 512 pts → 1 padded token/cell
cells with > 512 pts → 2+ tokens/cell (split by latitude sort)
N_tok_CERRA ≈ 1,600–3,000  (before masking, rough estimate)
```

**Token feature assembly** (`tokenize_apply_mask_source`):

Each token has shape `[token_size, F_token]` where:

```
F_token = 1 + 5 + 2 + 2 + F_s

components:
  stream_id       [1]   — constant scalar identifying the stream
  time_encoding   [5]   — (year/2100, doy/365, hour/1440, sin(Δt), cos(Δt))
  coords_local    [2]   — r3tos2( R·x_point ) : point position rotated to cell-center frame,
                          then projected to (θ, φ) ∈ R²
  geoinfos        [2]   — normalized orog, lsm
  data            [F_s] — normalized meteorological channels
```

| Stream | F_token | Approx |
|--------|---------|--------|
| ERA5   | 10 + F_s ≈ | **173** |
| CERRA  | 10 + 85 = | **95** |

**Masking semantics at tokenization level:**

The `mask_tokens` boolean array from the `Masker` filters which tokens are passed to the embedding.  
In forecasting mode (`masking_strategy: "forecast"`):
- Context region: cells **not masked** — their source tokens are assembled and passed to the encoder.
- Masked cells: their source tokens are **entirely absent** from `source_tokens_cells` (the cell's `cell_lens` entry is zero). The cells still exist in the global lattice but receive no local data — they are filled with the learnable query `q_cells + pe_global` in the encoder.
- `masking_rate = 0.6` → 60% of cells are masked as source; 40% (≈4,915 cells) contribute source tokens.
- `masking_rate_none = 0.05` → 5% are never predicted (just used as context).

**Memory at tokenization stage:**

```
ERA5 tokens (after masking):  4,915 tokens × 64 × 173 × 4 bytes ≈ 218 MB (CPU float32)
CERRA tokens (unmasked set): ~600 tokens × 512 × 95 × 4 bytes ≈ **116 MB** (CPU float32)
```

Stored in `StreamData.source_tokens_cells[step]` as `[N_tok, token_size, F_token]` float32.

**Gradient:** None (data loading, no autograd).

---

## Stage 2 — Stream Embedding

### Entry point: `EmbeddingEngine.forward` → `StreamEmbedTransformer.forward_channels`

**Configuration:**
```yaml
embed:
  net: transformer
  num_tokens: 1
  num_heads: 8
  dim_embed: 512     # internal dim
  num_blocks: 2
embed_orientation: channels
embed_unembed_mode: block
```

### Conceptual Architecture — **What is a "token"?**

The `embed_orientation = "channels"` mode implements **per-channel projection** (Option A hybrid):

```
Input token x_in: [N_tok, token_size, F_token]

Step 1 — Transpose:
  x_in.transpose(-2,-1) → [N_tok, F_token, token_size]
  
  Interpretation: each of the F_token channels becomes a sequence element;
  the T=token_size spatial samples are that channel's "signal" within the cell.

Step 2 — Linear embed (per channel independently):
  Linear(token_size → D_stream=512)
  → [N_tok, F_token, D_stream]

Step 3 — Harmonic positional encoding over F_token sequence:
  + pe: [F_token, D_stream]   (standard sinusoidal, position = channel index)
  → [N_tok, F_token, D_stream]

Step 4 — 2× (MultiSelfAttentionHead + MLP) internal transformer:
  Self-attention over F_token tokens, each of dim D_stream
  Attention complexity: O(F_token²) per token, F_token ≈ 173 ERA5 / 95 CERRA
  flash_attn_func (padded, not varlen — tokens packed as a batch)
  QK-norm: LayerNorm(D_stream / n_heads = 64)
  → [N_tok, F_token, D_stream]

Step 5 — Block unembedding:
  For each channel i: Linear(D_stream=512 → D_local/F_token)
    ERA5:  Linear(512 → 2048/173 ≈ 11)  [truncated; padded to exact multiple]
    CERRA: Linear(512 → 2048/95 ≈ 21)
  Stack+flatten: [N_tok, F_token, D_local/F_token] → [N_tok, D_local]

Step 6 — Reshape to num_tokens=1:
  [N_tok, 1, D_local=2048]
  Then .flatten(0,1) → [N_tok, D_local=2048]
```

**Output shapes:**

```
ERA5  embeddings: [N_tok_ERA5,  D_local]  =  [~4,915,  2048]  bfloat16
CERRA embeddings: [N_tok_CERRA, D_local]  =  [~600,    2048]  bfloat16
```

**Semantics:** Each token now encodes a single spatial "patch" of ~44 points (ERA5) or ~250–1700 points (CERRA) into a single 2048-dim latent vector. The `token_size` spatial samples are not individually preserved — they are averaged/combined by the linear projection + cross-channel transformer.

**Key insight:** This is **not** sequential tokenization. The T spatial samples in a token are treated as independent dimensions of a single channel measurement, not as a time sequence. Information about sub-cell spatial variation is partially captured by the latitude-sorted ordering (the linear sees the sorted signal), but there is no explicit positional encoding over the T points — **fine-scale sub-cell structure is compressed into 512/F_token ≈ 3–58 dimensions per channel**. This is the first information bottleneck.

**dtype:** Input sdata is cast to `tokens_all.dtype` (bfloat16). Internal computations in the stream transformer use float32 (PyTorch AMP) or bfloat16 depending on layer type. Flash attention uses bfloat16.

**Gradient:** Full gradients through all Linear and attention layers (no `no_grad` here).

**Memory (activation, training):**
```
Internal [N_tok, F_token, D_stream]:
  ERA5:  4915 × 173 × 512 × 2 bytes (bf16) ≈ 872 MB  (checkpointed → recomputed)
  CERRA: 600  × F_token_cerra × 512 × 2 bytes   (F_token_cerra = 10+F_s_cerra, unknown without zarr)
Output [N_tok, D_local]:
  (4915 + 600) × 2048 × 2 bytes              ≈  22 MB
```

Gradient checkpointing is applied via `checkpoint(self.embed, ...)` — activations are **not stored during forward**, recomputed during backward.

---

## Stage 3 — Cross-Stream Scatter + Local Positional Encoding

### Entry point: `EmbeddingEngine.forward` (continuation)

**Purpose:** Reorder tokens from stream-centric order to cell-centric order, so that all tokens from different streams that fall in the same HEALPix cell are contiguous.

```
tokens_all = torch.empty([N_tok_total, D_local=2048])  bfloat16

scatter_idxs computed from batch.tokens_lens:
  tokens_lens: [n_steps, B, n_streams=2, C]

Scatter ERA5 and CERRA embeddings into cell-ordered positions:
  tokens_all[scatter_idxs] = cat([era5_embeds, cerra_embeds])

→ tokens_all[cell j]:  n_tokens_j consecutive rows
   where n_tokens_j = n_ERA5_toks_j + n_CERRA_toks_j
```

**Local Positional Encoding** (`pe_embed`):

```
pe_embed: [MAX=64, D_local=2048]  non-trainable, stored as nn.Parameter(requires_grad=False)

Initialization: sinusoidal with bias (token_idx_bias=16, freq_bias=8):
  position ∈ [16, 79]  (offset to distinguish from global PE)
  pe_embed[:, 0::2] = sin(pos × exp(-freq_bias × log(64) / D_local))
  pe_embed[:, 1::2] = cos(...)

pe_idxs: within-cell token index (0-based) for each token
tokens_all = tokens_all + pe_embed[pe_idxs]
```

This PE distinguishes token #0 from token #1 within a cell (relevant for CERRA cells with multiple tokens). It does NOT encode the cell's spatial location — that is handled by `pe_global` later.

**Output:**

```
tokens_all: [N_tok_total, D_local=2048]  bfloat16
  N_tok_total ≈ N_tok_ERA5 + N_tok_CERRA ≈ 5,515 (rough estimate)

cell_lens: [C=12,288]  int32  — number of tokens per cell
  value 0: masked cell (no source data)
  value 1: one stream present (typically ERA5-only cells)
  value 2: two streams in cell (ERA5 + CERRA, only for European cells)
```

**Gradient:** Full gradients through scatter and PE addition. `pe_embed` has `requires_grad=False` (frozen).

---

## Stage 4 — Local Assimilation (0 Blocks — Identity Pass)

### Entry point: `LocalAssimilationEngine.forward`

**Config:** `ae_local_num_blocks: 0`

This is a **no-op** in the forecasting config. The module exists but has an empty `ae_local_blocks` ModuleList.

```
tokens_all → tokens_all  (unchanged)
Shape: [N_tok_total, D_local=2048]  bfloat16
```

**Design note:** When `ae_local_num_blocks > 0`, this would be:
- `ae_local_num_blocks` × (`MultiSelfAttentionHeadVarlen` + `MLP`) 
- Varlen self-attention: each cell is an independent sequence; tokens from different cells never attend to each other.
- Attention complexity per cell: O(n_tokens_j²) where n_tokens_j is typically 1–5 → negligible.
- This stage is where multi-stream, multi-token-per-cell information mixing would happen.

**Compression ratio here:**
- ERA5 cells: 44 points → 1 token of 2048 dims. Compression = `44 × 173 / 2048 ≈ 3.7×` (information-theoretically lossy).
- CERRA cells: up to 1700 points → 1–3 tokens. For a cell with 1700 points: `1700 × 95 / (3 × 2048) ≈ 26×` compression. **This is the primary bottleneck** for CERRA sharpness.

---

## Stage 5 — Local → Global Adapter

### Entry point: `Local2GlobalAssimilationEngine.forward` (called from `assimilate_local_project_chunked`)

**Config:**
```yaml
ae_local_num_queries: 1    # Q=1
ae_adapter_num_blocks: 2   # default
ae_adapter_num_heads: 16
ae_adapter_embed: 128      # dim_head_proj for cross-attention
```

**Learnable queries** (`q_cells`):

```
q_cells: [C=12,288, Q=1, D_global=2048]  float32  requires_grad=True

Initialization:
  Uniform([0, 1/D_global])  with embedded metadata:
  q_cells[:, :, -10:-9] = query_index
  q_cells[:, :, -9:-8]  = cell_index / C
  q_cells[:, :, -8:-6]  = cos(HEALPix theta) — elevation
  q_cells[:, :, -6:-3]  = sin(HEALPix phi)   — azimuth
  q_cells[:, :, -3:]    = query_index again
```

**Global PE** (`pe_global`):

```
pe_global: [C, Q=1, D_global=2048]  non-trainable

pe_global[cell, query, 0::2] = 0.5×sin(8×query_idx × xs) + sin(cell_idx × xs)
pe_global[cell, query, 1::2] = 0.5×cos(...) + cos(...)
where xs = 2π × arange(0, D/2) / D

Combined: tokens_global = q_cells + pe_global  → [rs, C, Q, D_global]
tokens_global.reshape(rs, C, Q×D_global).flatten(1,2) = [rs, C×Q, D_global]
```

**Critical role of pe_global:** Masked cells (60% of C) receive no local tokens and no cross-attention update — their representation is `q_cells[cell] + pe_global[cell]`. Without `pe_global`, all masked cells are identical (since `q_cells` is a shared parameter when `ae_local_queries_per_cell=False`). The sinusoidal PE provides spatial identity, preventing rank collapse.

**Chunked processing** (to work around Flash Attention bugs):

The adapter processes cells in chunks of `C / 2 = 6,144` (for healpix_level≤5) or `C / 8 = 1,536` (for level>5).

**Cross-attention block** (`MultiCrossAttentionHeadVarlenSlicedQ`):

```
For each unmasked chunk of cells:

Q: tokens_global_unmasked[cells_in_chunk]  [N_unmasked_chunk, Q=1, D_global=2048]  bf16
K: tokens_all[tokens_in_chunk]             [N_tok_chunk, D_local=2048]             bf16
V: same as K

QK projection: Linear(D_global → n_heads×dim_head_proj) = Linear(2048 → 16×128=2048)
K projection:  Linear(D_local  → n_heads×dim_head_proj) = Linear(2048 → 16×128=2048)

Attention: flash_attn_varlen_func
  Sequence structure: each cell is an independent group
  Q varlen: [N_unmasked_cells, 1] queries
  KV varlen: [N_tok_per_cell, ...] keys/values per cell
  → Output: [N_unmasked_cells, 1, D_global=2048]

With ae_adapter_num_blocks=2:
  Block 1: CrossAttn(Q=query, KV=local_tokens)
  Block 2: MLP(D_global=2048) + CrossAttn(Q=updated_query, KV=local_tokens)
```

**Attention complexity per cell:**
```
Local adapter:  O(Q × n_tokens_j)  per cell
  ERA5: O(1 × 1) = O(1)  (trivially cheap — single query attends to single token)
  CERRA: O(1 × n_cerra_tokens_j)  where n_cerra_tokens_j ≤ 3 for most cells
Total across all unmasked cells: O(N_unmasked × max_tokens_per_cell)
```

**After adapter:**

```
tokens_global_unmasked: [N_unmasked_total, Q=1, D_global=2048]  bfloat16
```

Then scatter-filled back into the full lattice:

```
tokens_global: [rs, C, Q, D_global] = [B, 12288, 1, 2048]
  masked cells: q_cells[cell] + pe_global[cell]  (frozen queries, no local data)
  unmasked cells: cross-attention output (updated queries)

Flatten: [B, C×Q, D_global] = [B, 12288, 2048]
```

**Aggregation engine** (`ae_aggregation_num_blocks: 0`): No-op in this config.

**Output of `assimilate_local`:**

```
tokens_global: [B, C×Q=12288, D_global=2048]  bfloat16
posteriors: scalar (0 — latent noise KL disabled)
```

**Gradient:** Full gradients through q_cells and all cross-attention parameters.  
`q_cells` gradient norm is critical to monitor — it controls how well masked cells learn positional identity vs. content.

---

## Stage 6 — Global Assimilation Transformer

### Entry point: `GlobalAssimilationEngine.forward`

**Config:**
```yaml
ae_global_num_blocks: 4
ae_global_num_heads: 32
ae_global_att_dense_rate: 1.0   # all blocks are global (dense) attention
ae_global_mlp_hidden_factor: 2
rope_2D: False                  # absolute PE only
```

**Architecture:** 4 × (`MultiSelfAttentionHead` + `MLP`)

```
Input:  tokens_global [B, 12288, D_global=2048]  bfloat16

For each of 4 blocks:

  Self-Attention (dense global):
    LN(tokens) → Q, K, V: Linear(2048 → 32×64=2048)
    QK-norm: LayerNorm(elementwise_affine=False, eps=1e-4) per head
    flash_attn_func: full sequence Q=K=12288, heads=32, head_dim=64
    + residual
    → [B, 12288, 2048]

  MLP:
    hidden = 2048 × hidden_factor=2 = 4096
    GELU(Linear(2048→4096)) → Linear(4096→2048) + residual
    + PreNorm (LN before MLP)
    → [B, 12288, 2048]

  (gradient checkpointing on every block via torch.utils.checkpoint)

Output: tokens_global [B, 12288, D_global=2048]  bfloat16
```

**Attention complexity:**
$$\text{FLOPs per block} \approx 4 \times B \times C^2 \times D_{\text{head}} \times n_{\text{heads}}$$
$$= 4 \times 1 \times 12288^2 \times 64 \times 32 \approx 390 \text{ GFLOPs}$$

With 4 blocks: **≈1.56 TFLOPs** for global attention alone per forward pass.

**Memory (KV cache during attention):**

```
Q, K, V each: [B, n_heads, C, head_dim] = [1, 32, 12288, 64]  bfloat16
  = 32 × 12288 × 64 × 2 bytes = 50 MB per QKV = 150 MB total
Flash attention: O(C) auxiliary memory (HBM), not O(C²)
```

**Normalization:** PreNorm with `LayerNorm(elementwise_affine=False, eps=1e-4)` — **no learned scale/shift**. Gradients flow through the normalized values only.

**Positional encoding during global attention:** When `rope_2D=False` (default), no positional encoding is applied in Q/K. The spatial positions of the 12,288 cells are encoded only through:
1. `pe_global` (absolute sinusoidal, cell-index-based, added at Stage 5)
2. The HEALPix cell ordering in NESTED scheme (no inherent spatial locality guarantee)

When `rope_2D=True`, `rotary_pos_emb_2d` is applied: HEALPix cell centers (lat, lon in radians) are used to compute 2D RoPE embeddings, giving relative spatial position awareness to Q/K.

**Gradient checkpointing:** `checkpoint(block, tokens, coords, aux_info, use_reentrant=False)` — activations for each block are discarded during forward and recomputed during backward. Reduces peak memory by ~4× at the cost of ~30% extra compute.

---

## Stage 7 — Multi-Step Reshape

### In `Model.forward`:

```python
shape = (B, n_steps, C*Q, D_global)
tokens = tokens.reshape(shape).sum(axis=1)
→ tokens: [B, C*Q=12288, D_global=2048]
```

When `n_steps=1`, this is identity. For multiple input steps (e.g., multi-step assimilation), the encodings are summed — a simple but potentially lossy operation that discards temporal ordering between steps.

---

## Stage 8 — Forecasting Engine

### Entry point: `ForecastingEngine.forward`

**Config:**
```yaml
fe_num_blocks: 16
fe_num_heads: 16
forecast_att_dense_rate: 1.0   # all blocks global dense
fe_layer_norm_after_blocks: [7]  # LayerNorm inserted after block index 7
fe_impute_latent_noise_std: 1e-4
forecast:
  time_step: 06:00:00
  offset: 1
  num_steps: 3
  policy: fixed
```

**Architecture:** 16 × (`MultiSelfAttentionHead` + `MLP`), with a `LayerNorm` after block 7.

```
Input per forecast step: tokens [B, 12288, D_global=2048]  bfloat16

Training noise imputation (only during training):
  tokens += randn_like(tokens) × ||tokens|| × 1e-4
  (stabilizes gradients through the forecasting steps)

For each of 16 blocks:
  Self-Attention (global dense):
    n_heads=16, head_dim=2048/16=128
    flash_attn_func: full Q=K=12288
    QK-norm: LayerNorm(128, elementwise_affine=False)
    → [B, 12288, 2048]
  
  MLP:
    [B, 12288, 2048] → [B, 12288, 4096] → [B, 12288, 2048]
  
  (After block 7): LayerNorm(2048, elementwise_affine=False)
  
  (gradient checkpointing per block)

Output per step: tokens [B, 12288, D_global=2048]  bfloat16
```

**Rollout loop** (`batch.get_output_idxs()` = [0, 1, 2] for 3 steps):

```
step 0: tokens_t0 → ForecastingEngine → tokens_t1 → decode → pred_t1
step 1: tokens_t1 → ForecastingEngine → tokens_t2 → decode → pred_t2
step 2: tokens_t2 → ForecastingEngine → tokens_t3 → decode → pred_t3
```

With `pushforward=False` (default): all steps contribute to the loss and gradients flow through all 3 × 16 = 48 transformer blocks.

**FLOPs per forecast step:** Same as global encoder (~390 GFLOPs/block × 16 blocks = **6.24 TFLOPs**).

**Memory:** Each forecast step generates activations; with gradient checkpointing, peak memory is dominated by the last step's activations ≈ same as encoder stage.

**Weight initialization:** Linear layers in forecast blocks initialized with `Normal(0, 0.001)` — very small initialization to avoid the forecasting engine dominating early training, allowing the encoder to stabilize first.

---

## Stage 9 — Decoder Queries (Target Coordinate Embedding)

### Entry point: `Model.predict_decoders`

**Target coordinates assembly** (`get_target_coords_local`):

Target points are sampled uniformly up to `max_num_targets`:
- ERA5: up to 270,000 target points
- CERRA: up to 570,000 target points

Each target point `p` receives a coordinate feature vector:

```
target_coord_features[p]: [dim_coord_in]

Components:
  stream_id       [1]    — stream identifier
  time_encoding   [5]    — sin/cos of relative time in window (encode_times_target)
  geoinfos        [2]    — normalized orog, lsm at target point
  local_coords    [75]   — 5 cell-corner × (3D R3 rotated to center) × 5 vertex transforms
                           computed as: R_corner × p_R3, each 3-dim → 5×15=75
  neighbor_ctrs   [24]   — 8 HEALPix neighbor cell centers × 3D R3 = 24

dim_coord_in = 1 + 5 + 2 + 75 + 24 = 107
```

**Target coordinate embedding:**

```
embed_target_coords[stream]: Linear(107 → D_coord=512, bias=False)

Input:  t_coords [N_targets_stream, 107]   float32
Output: tc_tokens [N_targets_stream, 512]  bfloat16
```

The `embed_target_coords` is a simple linear projection — no positional encoding, no nonlinearity. The full spatial geometric context (neighbor positions, cell corners) is directly baked into the 107-dim input.

---

## Stage 10 — Decoder: TargetPredictionEngineClassic (PerceiverIO)

### Entry point: `TargetPredictionEngineClassic.forward`

**Config:**
```yaml
decoder_type: PerceiverIOCoordConditioning
target_readout:
  num_layers: 2
  num_heads: 4
pred_self_attention: True
pred_mlp_adaln: True
```

**1-ring neighborhood KV construction:**

```python
idxs = model_params.hp_nbours  # [C, 9]: each cell's self + 8 HEALPix neighbors
tokens_nbors = tokens.reshape([B, C, Q, D])\
                     .flatten(0,1)[idxs.flatten()]\
                     .flatten(0,1)
# shape: [B×C×9×Q, D_global=2048]

tokens_nbors_lens: [B×C + 1]  — all entries = 9×Q=9
```

Each target point attends to exactly 9 × Q = 9 global latent vectors (the cell it falls in, plus its 8 neighbors).

**PerceiverIO cross-attention blocks:**

```
dims_embed = [512, 512]  (num_layers=2, so 3 dims of equal size)
num_layers=2 → 2 × (CrossAttn + SelfAttn + MLP) blocks

For each layer i ∈ {0, 1}:

  Cross-Attention (Q=tc_tokens, KV=tokens_nbors):
    dim_embed_q = 512 (target query)
    dim_embed_kv = 2048 (global latent)
    n_heads = 4, dim_head_proj = None → 512/4 = 128
    QK-norm: LayerNorm(128, elementwise_affine=False)
    aux = t_coords [N_targets, 107] → AdaLayerNorm conditioning
    flash_attn_varlen_func: varlen over N_targets, each attending to 9 KV tokens
    + residual
    → tc_tokens [N_targets, 512]

  Self-Attention (pred_self_attention=True):
    dim_embed = 512, n_heads = 4
    Over N_targets tokens (full batch, all target points of this stream)
    aux = t_coords → AdaLayerNorm
    flash_attn_varlen_func: grouped by sample
    → tc_tokens [N_targets, 512]

  MLP:
    512 → 1024 → 512
    AdaLayerNorm (pred_mlp_adaln=True): aux=t_coords→scale/shift
    → tc_tokens [N_targets, 512]
```

**Output:**
```
tc_tokens: [N_targets_stream, D_coord=512]  bfloat16
```

**Attention complexity:**

```
Cross-attention: O(N_targets × 9) — extremely cheap (fixed 9 KV)
Self-attention:  O(N_targets²)    — expensive for large N_targets

ERA5:  N_targets ≈ 270,000 (masked cells × pts ≈ 12,288 × 0.6 × 44)
  Self-attn: O(270,000²) ≈ 72.9 billion ops → NOT FEASIBLE without chunking/sparsity
```

> **Warning:** With `pred_self_attention=True` and `max_num_targets=270,000`, the self-attention in the decoder has quadratic cost O(N²). This is the dominant memory/compute bottleneck for large target sets. In practice, the actual number of used targets is bounded by `max_num_targets` and the varlen mechanism processes samples independently (each sample has `N_targets/B` targets). For batch size B=8 on 8 GPUs with FSDP, each GPU sees ~34k targets, giving O(34k²) ≈ 1.16B ops — feasible.

---

## Stage 11 — Prediction Head

### Entry point: `EnsPredictionHead.forward`

**Config:** `pred_head: {ens_size: 1, num_layers: 1}`

```
Input: tc_tokens [N_targets, D_coord=512]  bfloat16

1 ensemble member, 1 layer:
  Linear(512 → F_target)  — no normalization, no nonlinearity

Output: pred [ens_size=1, N_targets, F_target]
```

Where `F_target` is the number of target (output) channels:
- ERA5: all channels except those in `target_exclude` (removes `w_`, `slor`, `sdor`, `tcw`, `cp`, `tp`) ≈ **161 channels**
- CERRA: target channels minus `target_exclude` ≈ **22 channels**

The output is in normalized space — same normalization as the input. No learned denormalization; the loss is computed in normalized space.

**Final output per stream per forecast step:**

```
ERA5  pred: [1, N_targets_ERA5,  161]   bfloat16  (split into per-sample list)
CERRA pred: [1, N_targets_CERRA,  22]   bfloat16
```

Split by sample: `torch.split(pred, t_coords_lens, dim=1)` → list of B tensors.

---

## Stage 12 — Loss

### Entry point: Training loop / `LossPhysical`

**Loss function:** MSE in normalized space

```
loss_ERA5  = mean( (pred_ERA5  - target_ERA5 )² × channel_weights × location_weights )
loss_CERRA = mean( (pred_CERRA - target_CERRA)² × channel_weights × location_weights )
total_loss = 1.0 × loss_ERA5 + 1.0 × loss_CERRA
  (loss_weight: 1.0 for both streams)
```

**Location weights:** `location_weight: cosine_latitude` — each target point is weighted by cos(lat), upweighting equatorial regions. This counteracts the higher density of HEALPix cells near the equator (roughly uniform area weighting).

**Channel weights:** Per-variable weights from zarr statistics, typically set equal or proportional to inverse variance. Target channel weights from `parse_target_channel_weights()`.

**Gradient flow:**

```
loss ← pred_head ← tc_tokens ← TargetPredictionEngine(cross-attn) ←
        tokens_global ← ForecastingEngine (×3 steps) ←
        tokens_global ← GlobalAssimilation ←
        tokens_global ← Local2GlobalAdapter ←
        tokens_all ← StreamEmbed ← source_tokens_cells
```

`source_tokens_cells` tensors are leaf nodes (no gradients into the data). The loss gradient flows through all model parameters.

**Gradient checkpointing:** Every transformer block uses `torch.utils.checkpoint`, halving peak activation memory at the cost of ~30% extra compute (one additional forward pass per block during backward).

---

## Complete Tensor Shape Summary

```
Stage               Tensor                    Shape                      dtype
─────────────────────────────────────────────────────────────────────────────────
0. Raw data         era5_data                 [P_e, F_s_era5]            float32
                    cerra_data                [P_c, F_s_cerra]           float32

1. Tokenized        source_tokens_cells_era5  [N_tok_e, T_e=64,  F_e]    float32
                    source_tokens_cells_cerra [N_tok_c, T_c=512, F_c]    float32
                      N_tok_e ≈ 4,915  (40% of C after masking)
                      N_tok_c ≈ 600    (40% of CERRA cells)
                      F_e = 10 + F_s_era5 ≈ 173
                      F_c = 10 + 85 = 95

2. Embedded         era5_embeds               [N_tok_e, D_local=2048]    bf16
                    cerra_embeds              [N_tok_c, D_local=2048]    bf16

3. Scattered+PE     tokens_all                [N_tok_total≈5515, 2048]   bf16

4. Local attn       tokens_all (no-op)        [N_tok_total, 2048]        bf16

5. L2G adapter      tokens_global_unmasked    [N_unmasked≈4915, Q=1, 2048] bf16
   (masked fill)    tokens_global             [B, C×Q=12288, 2048]       bf16

6. Global attn      tokens_global             [B, 12288, 2048]           bf16

7. Reshape          tokens                    [B, 12288, 2048]           bf16

8. ForecastEngine   tokens (per step)         [B, 12288, 2048]           bf16

9. Query embed      tc_tokens_era5            [N_tgt_e, D_coord=512]     bf16
                    tc_tokens_cerra           [N_tgt_c, D_coord=512]     bf16
                      N_tgt_e ≤ 270,000
                      N_tgt_c ≤ 570,000

10. Decoder         tc_tokens_era5            [N_tgt_e, 512]             bf16
                    tc_tokens_cerra           [N_tgt_c, 512]             bf16

11. Pred head       pred_era5                 [1, N_tgt_e, F_tgt_era5≈161] bf16
                    pred_cerra                [1, N_tgt_c, F_tgt_cerra≈22] bf16

12. Loss            scalar                    []                         float32
```

---

## Normalization Strategy

| Where | Type | Per-what | Notes |
|-------|------|----------|-------|
| Input data | z-score: `(x-μ)/σ` | Per variable, globally over training set | Applied by anemoi dataset. μ,σ from 1979–2022 ERA5; 1985–2022 CERRA |
| Geoinfo (orog, lsm) | z-score | Per geoinfo channel | Separate mean/std from `ds.statistics` |
| Time encoding (source) | See below | — | year/2100, doy/365, hour/1440, then sin/cos |
| Time encoding (target) | sin/cos relative | — | Only relative time within window (+0.5 offset) |
| Target coords | Raw geometry | — | HEALPix R3 positions; no explicit normalization |
| Within stream embedder | Harmonic PE | — | Standard sinusoidal over channel index |
| Attention QK | LayerNorm per head | Per head-dim | `elementwise_affine=False`, eps=1e-4 |
| Layer inputs (global/forecast/decoder) | PreNorm LayerNorm | Per feature | `elementwise_affine=False` — no learnable params in norm |
| MLP with AdaLN (decoder) | AdaLayerNorm | Per feature, conditioned on t_coords | Scale/shift computed from 107-dim coordinate features |
| Loss | In normalized space | — | MSE on z-scored predictions |
| No seasonal normalization | — | — | Not implemented; model must learn seasonal patterns from time encoding |
| No RMS or per-level normalization | — | — | All levels pooled into same global statistics |

---

## Masking Semantics (Exact)

**What "masked" means in WeatherGenerator:**

Masking operates at the **HEALPix cell level**, not the token or point level.

```
mask = [True/False] × C cells

Masked cell (mask=False):
  - source_tokens_cells[step][cell_idx] = None  (cell gets 0 tokens)
  - cell_lens[cell_idx] = 0
  - In encoder: cell uses q_cells[cell] + pe_global[cell] as latent (untouched by cross-attention)
  - Visible to global attention: YES (all C cells participate in global self-attention)
  - Predicted as target: YES (target coords are sampled from masked cells' grid points)

Unmasked cell (mask=True):
  - Gets 1+ tokens in source_tokens_cells
  - cell_lens[cell_idx] > 0
  - Local tokens → cross-attention with q_cells → updated global latent
  - May or may not be a prediction target
```

**Masking does NOT:**
- Zero the attention keys/values — masked cells attend normally in global transformers
- Replace tokens with a mask token — there are no mask tokens; cells are simply absent
- Block information flow — masked cells see context from unmasked cells via global attention

**Masking DOES:**
- Remove the data from those cells as source input
- Force the model to predict target variables at those cells from context only

**Masking strategy in forecasting mode:**

```
masking_strategy: "forecast"
masking_rate: 0.6       → 60% of C cells are masked as source
masking_rate_none: 0.05 → 5% cells: neither source nor target (dropout)
effective context: ~35% of C cells = ~4,300 cells
effective targets: ~60% of C cells × pts_per_cell
```

**Spatial structure:** In "forecast" mode, the masking is essentially the target grid (future state) vs. context grid (current state). The masking is **globally random per batch sample** — no spatial contiguity constraint in this mode (unlike JEPA's `cropping_healpix` strategy which maintains spatially contiguous patches).

---

## Positional Encoding Details

### Three PE systems used simultaneously:

**1. Local PE (`pe_embed`):** Applied after stream embedding, before local attention.
```
Scope: within-cell token index
Shape: [MAX_LOCAL=64, D_local=2048]
Type: sinusoidal with offset (token_idx_bias=16, freq_bias=8)
Purpose: distinguish token #0 from token #1+ in the same cell
Trainable: No
```

**2. Global PE (`pe_global`):** Added to learnable queries before local→global adapter.
```
Scope: cell identity in the global lattice
Shape: [C=12288, Q=1, D_global=2048]
Type: compound sinusoidal: sin/cos(cell_idx × xs) + 0.5×sin/cos(query_idx × xs)
Purpose: give each cell a unique spatial identity in global token sequence
  Critical for masked cells — their entire latent is Q+PE, no data signal
Trainable: No
```

**3. 2D RoPE (`rope_coords`, optional, default OFF):**
```
Scope: relative position in global/forecast/aggregation attention
Shape: [1, C×Q=12288, 2]  → cos,sin of shape [12288, D_head]
Type: 2D rotary from HEALPix cell centers (lat, lon) in radians
  lat,lon → inv_freq → (freq_lat, freq_lon) → cat → cos,sin
Applied to Q,K (not V) via: q_embed = q*cos + rotate_half(q)*sin
Trainable: No
Polar treatment: Cell centers at poles have lat≈±π/2; RoPE
  computes cos/sin of these values normally — no special treatment.
  The non-uniform cell area of HEALPix means polar cells cover more area,
  but the PE encodes only center position, not area.
```

**When `rope_2D=False`:** The absolute pe_global encodes position, but this is a fixed lookup into a sinusoidal table indexed by cell number (in NESTED ordering). The NESTED ordering has locality properties (nearby cells in space tend to have nearby indices for the same sub-level), but it is **not** rotationally equivariant. The model must learn geographic structure from scratch.

---

## Attention Complexity and FLOPs Summary

| Module | Sequence Length | Heads | Head Dim | FLOPs (per block) | Memory (KV) |
|--------|----------------|-------|----------|------------------|-------------|
| Stream embed internal | F_e≈173 or F_c=95 | 8 | 64 | O(173²)≈0.3M / O(95²)≈0.9M per token | negligible |
| Local attention (0 blocks) | n_tok/cell ≈1–3 | 16 | 128 | 0 (disabled) | 0 |
| L2G cross-attention | 1 Q, ≤3 KV | 16 | 128 | O(1×3) per cell | negligible |
| Aggregation (0 blocks) | N_unmasked≈4,915 | 32 | 64 | 0 (disabled) | 0 |
| **Global encoder** (4 blocks) | **12,288** | **32** | **64** | **≈390 GFLOPs** | **150 MB** |
| **Forecast engine** (16 blocks) | **12,288** | **16** | **128** | **≈390 GFLOPs** | **150 MB** |
| Decoder cross-attn (2 layers) | 9 KV, N_tgt Q | 4 | 128 | O(N_tgt × 9) ≈ cheap | negligible |
| Decoder self-attn (2 layers) | N_tgt≈34k/GPU | 4 | 128 | O(34k²) ≈ 1.16B | 68 MB |

**Total FLOPs per forward pass (rough estimate):**
```
Stream embedding:        ~50 GFLOPs
L2G adapter:             ~5 GFLOPs
Global encoder (4 blocks): ~1.56 TFLOPs
Forecast engine (16 blocks × 3 steps): ~18.7 TFLOPs
Decoder (2 layers per stream): ~500 GFLOPs
──────────────────────────────────────
Total: ~21 TFLOPs per training step
```

---

## Information Bottlenecks

### Bottleneck 1 — ERA5 Token Compression
```
Input:   44 points × 173 features/point = 7,612 scalars per cell
Output:  1 latent vector of 2048 scalars
Ratio:   7,612 / 2048 ≈ 3.7×

Mechanism: Linear(T=64 → D_stream=512) per channel, then inter-channel transformer
Sub-cell structure preserved: partial — the T=44 values are compressed by a linear
projection, preserving any linear combination. The latitude sort provides ordering
but no distance-based structure.
```

### Bottleneck 2 — CERRA Extreme Compression
```
Input:   up to 1700 points × 95 features/point = 161,500 scalars per cell
Output:  1–3 tokens × 2048 = 2,048–6,144 scalars
Ratio:   up to 161,500 / 2,048 ≈ 79×  (dense cell); ~26× for 3-token cells

This is where CERRA sharpness is most threatened.
The linear embedding projects each of 95 channels' 512 spatial values
into 2048/95 ≈ 21 output dims per channel.
Small-scale European precipitation features (~5km) that occur within
a single HEALPix cell (~25km) cannot be reconstructed at the decoder
unless they project strongly onto the linear embedding.
```

### Bottleneck 3 — Masked Cell Imputation
```
60% of global cells have no source data.
Their representation is entirely q_cells[cell] + pe_global[cell]:
  - q_cells is either shared (ae_local_queries_per_cell=False) or per-cell
  - pe_global provides spatial identity but no weather-state information
  - Global attention must "fill in" missing cells from context alone
  
This is by design (JEPA-style: predict masked from context).
For forecasting, this means the model cannot simply "copy" the analysis
at masked cells — it must genuinely infer them from unmasked neighbors.
```

### Bottleneck 4 — Decoder 1-Ring KV
```
Each target point attends to 9 HEALPix cell latents (1-ring).
At HEALPix level 5, each cell is ~25km across.
The decoder has access to information from a ~75km radius (3 cells).
Finer-scale patterns must already be encoded in the cell latent.
ae_local_num_queries=1 means exactly ONE latent vector per cell
enters the decoder — no multi-query sub-cell representation.
```

---

## Data Sampling Strategy

### Temporal Sampling (`multi_stream_data_sampler.py`, `TimeWindowHandler`)

```
Training range:   1979-01-01 to 2022-12-31
Validation range: 2023-10-01 to 2023-12-31

time_window_step: 6h  → valid start times every 6h
time_window_len:  6h  → each sample is a 6h window

Total training timesteps: (2022-2022+43.8yr) × 365.25 × 4 ≈ 63,800 windows
```

**Sampling within mini-epoch:**
```
samples_per_mini_epoch: 4,096  training / 256 validation
num_mini_epochs: 64

Each mini-epoch: 4,096 timesteps drawn from the full training range.
shuffle: True → random temporal order within mini-epoch.
```

**No seasonal balancing or storm oversampling** in the default config.  
Uniform random sampling → winter/summer/equinoxes equally likely.  
Rare extremes (storms, heatwaves) are naturally under-sampled relative to their
importance — a known limitation of i.i.d. sampling for weather forecasting.

**CERRA temporal alignment:** CERRA at `frequency: 6h` is synchronized to ERA5.  
When CERRA data is missing for a timestamp, `DataReaderAnemoi` returns an empty  
`ReaderData` (0 points), effectively treating that time step as CERRA-absent.

**Geographic balancing:** None explicit. `location_weight: cosine_latitude` in the loss  
provides implicit geographic weighting — equatorial errors count less per unit area,  
compensating for the equal-area HEALPix tiling (which would otherwise over-emphasize  
tropics due to higher grid-point density there in ERA5's reduced Gaussian grid).

**Curriculum learning:** None. Masking rate is fixed at 60% throughout training.  
No progressive increase in context fraction, no difficulty scheduling.

---

## Gradient and Training Notes

### Gradient flow by parameter group

| Parameter | Shape | Trainable | Grad path |
|-----------|-------|-----------|-----------|
| `embed_engine.embeds[ERA5]` | ~10M params | Yes | tokens_all → loss |
| `embed_engine.embeds[CERRA]` | ~2M params | Yes | tokens_all → loss |
| `encoder.q_cells` | [12288, 1, 2048] | Yes | tokens_global → loss |
| `encoder.ae_local_global_engine` | ~100M params | Yes | tokens_global → loss |
| `encoder.ae_global_engine` (4 blocks) | ~200M params | Yes | tokens_global → loss |
| `forecast_engine` (16 blocks) | ~800M params | Yes | tokens → loss |
| `embed_target_coords[ERA5/CERRA]` | ~55k params | Yes | tc_tokens → loss |
| `target_token_engines[ERA5/CERRA]` | ~5M params/stream | Yes | tc_tokens → loss |
| `pred_heads[ERA5/CERRA]` | ~330k/33k params | Yes | pred → loss |
| `pe_embed`, `pe_global`, `hp_nbours` | fixed | No | — |

### Key optimizer settings
```
AdamW: β1=0.98125, β2=0.9875, ε=2e-8, weight_decay=0.1
  (scaled for multi-GPU: these are post-DDP-scaling values)
grad_clip: 1.0  (global L2 norm clipping)
lr schedule: warmup (256 steps) → cosine to 5e-5 → constant → linear cooldown
```

### FSDP sharding
With `with_fsdp: True`, parameters are sharded across GPUs. Gradients are reduced  
(all-reduce) across shards. The `mixed_precision_dtype: bf16` applies to forward/backward  
activations; master weights remain float32.
