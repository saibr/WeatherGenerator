# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import copy

import torch
from astropy_healpix import healpy
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.model.engines import (
    EmbeddingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    Local2GlobalSumEngine,
    LocalAssimilationEngine,
    QueryAggregationEngine,
)

# from weathergen.model.model import ModelParams
from weathergen.model.parametrised_prob_dist import LatentInterpolator
from weathergen.model.positional_encoding import positional_encoding_harmonic
from weathergen.utils.distributed import is_root


class EncoderModule(torch.nn.Module):
    name: "EncoderModule"

    def __init__(self, cf: Config, sources_size, targets_num_channels, targets_coords_size) -> None:
        """
        Initialize the EmbeddingEngine with the configuration.

        :param cf: Configuration object containing parameters for the engine.
        :param sources_size: List of source sizes for each stream.
        :param stream_names: Ordered list of stream identifiers aligned with cf.streams.
        """
        super(EncoderModule, self).__init__()
        self.cf = cf

        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level

        self.cf = cf
        self.sources_size = sources_size
        self.targets_num_channels = targets_num_channels
        self.targets_coords_size = targets_coords_size

        self.ae_aggregation_engine: QueryAggregationEngine | None = None
        self.ae_global_engine: GlobalAssimilationEngine | None = None
        self.ae_local_engine: LocalAssimilationEngine | None = None
        self.ae_local_global_engine: Local2GlobalAssimilationEngine | None = None
        self.embed_engine: EmbeddingEngine | None = None
        self.interpolator_latents: LatentInterpolator | None = None

        # embedding engine
        # determine stream names once so downstream components use consistent keys
        self.stream_names = list(cf.streams.keys())
        # separate embedding networks for differnt observation types
        self.embed_engine = EmbeddingEngine(cf, self.sources_size)

        assert cf.ae_global_att_dense_rate == 1.0, "Local attention not adapted for register tokens"
        self.num_register_tokens = cf.num_register_tokens
        self.num_class_tokens = cf.num_class_tokens

        # local assimilation engine
        self.ae_local_engine = LocalAssimilationEngine(cf)

        if cf.latent_noise_kl_weight > 0.0:
            self.interpolator_latents = LatentInterpolator(
                gamma=cf.latent_noise_gamma,
                dim=cf.ae_local_dim_embed,
                use_additive_noise=cf.latent_noise_use_additive_noise,
                deterministic=cf.latent_noise_deterministic_latents,
            )

        # local -> global assimilation engine adapter
        ae_adapter_type = cf.get("ae_adapter_type", "cross_attention")
        if ae_adapter_type == "sum":
            self.ae_local_global_engine = Local2GlobalSumEngine(cf)
        else:
            self.ae_local_global_engine = Local2GlobalAssimilationEngine(cf)

        # learnable queries
        if cf.ae_local_queries_per_cell:
            s = (self.num_healpix_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
            # add meta data
            q_cells[:, :, -8:-6] = (
                (torch.arange(self.num_healpix_cells) / self.num_healpix_cells)
                .unsqueeze(1)
                .unsqueeze(1)
                .repeat((1, cf.ae_local_num_queries, 2))
            )
            #theta, phi = healpy.pix2ang(
            #    nside=2**self.healpix_level, ipix=torch.arange(self.num_healpix_cells)
            #)
            theta, phi = healpy.pix2ang(
                nside=2**self.healpix_level, ipix=torch.arange(self.num_healpix_cells).cpu().numpy()
            )
            q_cells[:, :, -6:-3] = (
                torch.cos(theta).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -3:] = (
                torch.sin(phi).unsqueeze(1).unsqueeze(1).repeat((1, cf.ae_local_num_queries, 3))
            )
            q_cells[:, :, -9] = torch.arange(cf.ae_local_num_queries)
            q_cells[:, :, -10] = torch.arange(cf.ae_local_num_queries)
        else:
            s = (1, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            q_cells = torch.rand(s, requires_grad=True) / cf.ae_global_dim_embed
        self.q_cells = torch.nn.Parameter(q_cells, requires_grad=True)

        # query aggregation engine
        self.ae_aggregation_engine = QueryAggregationEngine(cf, self.num_healpix_cells)

        # global assimilation engine
        self.ae_global_engine = GlobalAssimilationEngine(cf, self.num_healpix_cells)

    def forward(self, model_params, batch):
        """
        Encoder forward
        """

        stream_cell_tokens = checkpoint(
            self.embed_engine, batch, model_params.pe_embed, use_reentrant=False
        )

        tokens_global, posteriors = checkpoint(
            self.assimilate_local, model_params, stream_cell_tokens, batch, use_reentrant=False
        )

        stream_diag = self._compute_stream_only_global_tokens_for_diag(
            model_params, stream_cell_tokens, batch, "ERA5"
        )
        if stream_diag is not None:
            self._print_stream_cosine_diag(
                "ae_adapter", tokens_global, stream_diag["tokens_global"], stream_diag["mask"]
            )

        tokens_global = checkpoint(
            self.ae_global_engine,
            tokens_global,
            coords=model_params.rope_coords,
            use_reentrant=False,
        )

        if stream_diag is not None:
            fork_devices = [tokens_global.device.index] if tokens_global.is_cuda else []
            with torch.random.fork_rng(devices=fork_devices):
                with torch.no_grad():
                    tokens_global_stream = self.ae_global_engine(
                        stream_diag["tokens_global"], coords=model_params.rope_coords
                    )
            self._print_stream_cosine_diag(
                "ae_global", tokens_global, tokens_global_stream, stream_diag["mask"]
            )

        return tokens_global, posteriors

    def _should_run_stream_similarity_diag(self, last_step_attr: str) -> bool:
        train_logging = self.cf.get("train_logging", {})
        interval = train_logging.get(
            "stream_similarity_interval",
            train_logging.get("cosine_similarity_interval", 1),
        )
        # ALWAYS default to 1 step interval if misconfigured
        interval = max(int(interval), 1)

        # IMPORTANT: do NOT rely on cf.general.istep (often stale / not updated)
        # Instead use a local counter on the module
        if not hasattr(self, "_diag_step"):
            self._diag_step = 0

        self._diag_step += 1

        if self._diag_step % interval != 0:
            return False

        return True

    def _compute_stream_only_global_tokens_for_diag(
        self, model_params, tokens: torch.Tensor, batch: ModelBatch, stream_name: str
    ):
        if not self._should_run_stream_similarity_diag(
            "_last_adapter_global_similarity_diag_step"
        ):
            return None

        stream_names = list(self.cf.streams.keys())
        if stream_name not in stream_names:
            return None

        stream_idx = stream_names.index(stream_name)
        tokens_lens_by_stream = batch.tokens_lens.permute([2, 0, 1, 3]).flatten(1, -1)
        stream_counts = tokens_lens_by_stream[stream_idx]
        if stream_counts.sum() == 0:
            return None

        full_counts = tokens_lens_by_stream.sum(0)
        prev_counts = tokens_lens_by_stream[:stream_idx].sum(0)
        max_tokens = full_counts.max()
        if max_tokens == 0:
            return None

        rows = torch.arange(max_tokens, device=tokens.device).unsqueeze(0)
        valid = (rows >= prev_counts.unsqueeze(1)) & (
            rows < (prev_counts + stream_counts).unsqueeze(1)
        )
        cell_offsets = torch.cat(
            [
                torch.zeros(1, device=tokens.device, dtype=torch.int64),
                full_counts.cumsum(0)[:-1].to(torch.int64),
            ]
        )
        idxs = (cell_offsets.unsqueeze(1) + rows).to(torch.int64)[valid]
        if idxs.numel() == 0:
            return None

        stream_tokens_lens = torch.zeros_like(batch.tokens_lens)
        stream_tokens_lens[:, :, stream_idx, :] = batch.tokens_lens[:, :, stream_idx, :]
        stream_batch = copy.copy(batch)
        stream_batch.tokens_lens = stream_tokens_lens

        rs = batch.get_num_steps() * len(batch)
        cell_mask = stream_counts.reshape(rs, self.num_healpix_cells).to(torch.bool)
        num_extra_tokens = self.num_register_tokens + self.num_class_tokens
        if num_extra_tokens > 0:
            extra_mask = torch.zeros(
                rs, num_extra_tokens, device=tokens.device, dtype=torch.bool
            )
            cell_mask = torch.cat([extra_mask, cell_mask], dim=1)
        token_mask = cell_mask.repeat_interleave(self.q_cells.shape[-2], dim=1).flatten()

        fork_devices = [tokens.device.index] if tokens.is_cuda else []
        with torch.random.fork_rng(devices=fork_devices):
            with torch.no_grad():
                tokens_global_stream, _ = self.assimilate_local(
                    model_params, tokens.detach()[idxs], stream_batch, enable_similarity_diag=False
                )

        return {"tokens_global": tokens_global_stream, "mask": token_mask}

    def _print_stream_cosine_diag(
        self, name: str, tokens_full: torch.Tensor, tokens_stream: torch.Tensor, token_mask: torch.Tensor
    ) -> None:
        if token_mask.sum() == 0:
            return
        if not is_root():
            return

        full_flat = tokens_full.reshape(-1, tokens_full.shape[-1])[token_mask].detach().float().flatten()
        stream_flat = (
            tokens_stream.reshape(-1, tokens_stream.shape[-1])[token_mask].detach().float().flatten()
        )
        denom = full_flat.norm() * stream_flat.norm()
        cos_sim = torch.dot(full_flat, stream_flat) / denom.clamp_min(torch.finfo(denom.dtype).eps)
        print(f"{name} full-vs-ERA5-only cos_sim = {cos_sim.item():.3e}")

    def interpolate_latents(self, tokens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ "
        TODO
        """

        if self.cf.latent_noise_kl_weight > 0.0:
            tokens, posteriors = self.interpolator_latents.interpolate_with_noise(
                tokens, sampling=self.stage
            )
        else:
            posteriors = torch.zeros((1,), device=tokens.device)

        return tokens, posteriors

    def assimilate_local_project_chunked(
        self, tokens, tokens_global, cell_lens, q_cells_lens, tokens_lens=None
    ):
        """
        Apply the local assimilation engine and then the
        local-to-global adapter using a chunking in the number of tokens
        to work around to bug in flash attention, the computations is performed in chunks
        """

        # combined cell lens for all tokens in batch across all input steps
        zero_pad = torch.zeros(1, device=tokens.device, dtype=torch.int32)

        # subdivision factor for required splitting
        clen = self.num_healpix_cells // (2 if self.cf.healpix_level <= 5 else 8)
        tokens_global_unmasked = []
        posteriors = []
        local_similarity_diag = self._init_ae_local_stream_similarity_diag(tokens_lens)

        for i in range(cell_lens.shape[0] // clen):
            # make sure we properly catch all elements in last chunk
            i_end = (i + 1) * clen if i < (cell_lens.shape[0] // clen) - 1 else cell_lens.shape[0]
            l0, l1 = (
                (0 if i == 0 else cell_lens[: i * clen].cumsum(0)[-1]),
                cell_lens[:i_end].cumsum(0)[-1],
            )

            toks = tokens[l0:l1]
            # if we have a very sparse input, we may have no tokens in the chunk, toks
            # skip processing of the empty chunk in this case
            # Check if this chunk is empty
            if l0 == l1 or toks.shape[0] == 0:
                continue

            toks_global = tokens_global[i * clen : i_end]
            cell_lens_cur = torch.cat([zero_pad, cell_lens[i * clen : i_end]])
            q_cells_lens_cur = q_cells_lens[: cell_lens_cur.shape[0]]

            # local assimilation model
            toks_in = toks
            toks = self.ae_local_engine(toks, cell_lens_cur, use_reentrant=False)
            self._accumulate_ae_local_stream_similarity_diag(
                local_similarity_diag, toks_in, toks, i * clen, i_end
            )

            toks, posteriors_c = self.interpolate_latents(toks)
            posteriors += [posteriors_c]

            # create mask for global tokens, without first element (used for padding)
            mask = cell_lens_cur[1:].to(torch.bool)
            toks_global_unmasked = toks_global[mask]
            q_cells_lens_unmasked = torch.cat([zero_pad, q_cells_lens_cur[1:][mask]])
            cell_lens_unmasked = torch.cat([zero_pad, cell_lens_cur[1:][mask]])

            # local to global adapter engine
            toks_global_unmasked = self.ae_local_global_engine(
                toks,
                toks_global_unmasked,
                q_cells_lens_unmasked,
                cell_lens_unmasked,
            )

            tokens_global_unmasked += [toks_global_unmasked]

        if len(tokens_global_unmasked) == 0:
            assert False, "Not yet implemented"
        tokens_global_unmasked = torch.cat(tokens_global_unmasked)
        self._print_ae_local_stream_similarity_diag(local_similarity_diag)

        return tokens_global_unmasked, posteriors

    def _init_ae_local_stream_similarity_diag(self, tokens_lens):
        if not self._should_run_stream_similarity_diag("_last_ae_local_similarity_diag_step"):
            return None

        stream_names = list(self.cf.streams.keys())
        if tokens_lens is None or "ERA5" not in stream_names:
            return None

        return {
            "stream_name": "ERA5",
            "stream_idx": stream_names.index("ERA5"),
            "tokens_lens_by_stream": tokens_lens.permute([2, 0, 1, 3]).flatten(1, -1),
            "dot": None,
            "full_norm_sq": None,
            "stream_norm_sq": None,
            "num_tokens": 0,
        }

    def _accumulate_ae_local_stream_similarity_diag(
        self, diag, tokens_full_in, z_full, cell_start, cell_end
    ) -> None:
        if diag is None:
            return

        stream_idx = diag["stream_idx"]
        chunk_counts = diag["tokens_lens_by_stream"][:, cell_start:cell_end]
        stream_counts = chunk_counts[stream_idx]
        if stream_counts.sum() == 0:
            return

        full_counts = chunk_counts.sum(0)
        prev_counts = chunk_counts[:stream_idx].sum(0)
        max_tokens = full_counts.max()
        if max_tokens == 0:
            return

        rows = torch.arange(max_tokens, device=tokens_full_in.device).unsqueeze(0)
        valid = (rows >= prev_counts.unsqueeze(1)) & (
            rows < (prev_counts + stream_counts).unsqueeze(1)
        )
        cell_offsets = torch.cat(
            [
                torch.zeros(1, device=tokens_full_in.device, dtype=torch.int64),
                full_counts.cumsum(0)[:-1].to(torch.int64),
            ]
        )
        idxs = (cell_offsets.unsqueeze(1) + rows).to(torch.int64)[valid]
        if idxs.numel() == 0:
            return

        stream_cell_lens = torch.cat(
            [
                torch.zeros(1, device=tokens_full_in.device, dtype=torch.int32),
                stream_counts.to(torch.int32),
            ]
        )
        fork_devices = [tokens_full_in.device.index] if tokens_full_in.is_cuda else []

        with torch.random.fork_rng(devices=fork_devices):
            with torch.no_grad():
                z_stream_only = self.ae_local_engine(
                    tokens_full_in.detach()[idxs], stream_cell_lens, use_reentrant=False
                )
                z_full_stream = z_full.detach()[idxs].float().flatten()
                z_stream_only = z_stream_only.float().flatten()
                dot = torch.dot(z_full_stream, z_stream_only)
                full_norm_sq = torch.dot(z_full_stream, z_full_stream)
                stream_norm_sq = torch.dot(z_stream_only, z_stream_only)

        diag["dot"] = dot if diag["dot"] is None else diag["dot"] + dot
        diag["full_norm_sq"] = (
            full_norm_sq
            if diag["full_norm_sq"] is None
            else diag["full_norm_sq"] + full_norm_sq
        )
        diag["stream_norm_sq"] = (
            stream_norm_sq
            if diag["stream_norm_sq"] is None
            else diag["stream_norm_sq"] + stream_norm_sq
        )
        diag["num_tokens"] += idxs.numel()

    def _print_ae_local_stream_similarity_diag(self, diag) -> None:
        if diag is None or diag["num_tokens"] == 0:
            return
        if not is_root():
            return

        denom = torch.sqrt(diag["full_norm_sq"]) * torch.sqrt(diag["stream_norm_sq"])
        cos_sim = diag["dot"] / denom.clamp_min(torch.finfo(denom.dtype).eps)
        print(
            f"ae_local full-vs-{diag['stream_name']}-only cos_sim = "
            f"{cos_sim.item():.3e} ({diag['num_tokens']} tokens)"
        )

    def aggregation_engine_unmasked(
        self,
        tokens_global_unmasked,
        tokens_global_register_class,
        tokens_lens,
        rope_cell_coords=None,
    ):
        """
        Aggregation engine on the global latents of unmasked cells
        """

        zero_pad = torch.zeros(1, device=tokens_global_unmasked.device, dtype=torch.int32)

        # tokens_global_unmasked: (total_unmasked, num_queries, global_dim)
        cell_lens_unflattened = torch.sum(tokens_lens, 2)
        cell_mask = cell_lens_unflattened.to(torch.bool)
        batch_lens = cell_mask.sum(dim=-1).flatten()
        expected_len = batch_lens.sum().item()
        actual_len = tokens_global_unmasked.shape[0]
        assert expected_len == actual_len, (
            f"Shape mismatch: expected {expected_len}, got {actual_len}"
        )
        # Flatten query dim into embed dim: (total_unmasked, num_queries * global_dim).
        # For num_queries=1 this is identical to the previous squeeze(0).
        # assimilate_local undoes this via reshape(..., q_c_shape[-2], q_c_shape[-1]).
        tokens_global_unmasked = tokens_global_unmasked.flatten(1, 2)
        if self.num_register_tokens + self.num_class_tokens > 0:
            assert self.cf.ae_local_num_queries == 1, (
                "ae_aggregation with register/class tokens and ae_local_num_queries > 1 "
                "is not yet supported"
            )
            tokens_global_unmasked = torch.split(tokens_global_unmasked, list(batch_lens))
            tokens_global_unmasked = torch.cat(
                [
                    t
                    for tup in zip(tokens_global_register_class, tokens_global_unmasked, strict=False)
                    for t in tup
                ],
                dim=0,
            )

        # Build packed coords matching the interleaved token order
        if rope_cell_coords is not None:
            num_extra = self.num_class_tokens + self.num_register_tokens
            zero_coords = torch.zeros(
                num_extra, 2, device=rope_cell_coords.device, dtype=rope_cell_coords.dtype
            )
            packed_coords = []
            for mask_b in cell_mask.flatten(0, 1):
                packed_coords.append(zero_coords)
                packed_coords.append(rope_cell_coords[mask_b])
            packed_coords = torch.cat(packed_coords, dim=0)
        else:
            packed_coords = None

        batch_lens = batch_lens + (self.num_class_tokens + self.num_register_tokens)
        batch_lens_patched = torch.cat([zero_pad, batch_lens], dim=0)
        assert (
            len(self.ae_aggregation_engine.ae_aggregation_blocks) == 0
            or self.cf.ae_local_num_queries == 1
        ), "ae_aggregation_num_blocks > 0 with ae_local_num_queries > 1 is not yet supported"
        tokens_global_unmasked = self.ae_aggregation_engine(
            tokens_global_unmasked, batch_lens_patched, use_reentrant=False, coords=packed_coords
        )

        return tokens_global_unmasked

    def assimilate_local(
        self,
        model_params,
        tokens: torch.Tensor,
        batch: ModelBatch,
        enable_similarity_diag: bool = True,
    ) -> torch.Tensor:
        """
        Processes embedded tokens locally and prepares them for the global assimilation

        Args:
            model_params : Query and embedding parameters
            tokens : Input tokens to be processed by local assimilation
            cell_lens : Used to identify range of tokens to use from generated tokens in cell
                embedding
        Returns:
            Tokens for global assimilation
        """

        cell_lens = torch.sum(batch.tokens_lens, 2).flatten()

        num_steps_input = batch.get_num_steps()
        rs = num_steps_input * len(batch)

        # create register and latent tokens and prepend to latent spatial tokens
        num_extra_tokens = self.num_register_tokens + self.num_class_tokens
        pos_enc = positional_encoding_harmonic
        tokens_global_register_class = pos_enc(self.q_cells.repeat(rs, num_extra_tokens, 1))

        # TODO: re-enable or remove ae_local_queries_per_cell
        if self.cf.ae_local_queries_per_cell:
            tokens_global = (self.q_cells + model_params.pe_global).repeat(rs, 1, 1)
        else:
            num_tokens = self.num_healpix_cells
            tokens_global = self.q_cells.repeat(num_tokens, 1, 1) + model_params.pe_global
            tokens_global = tokens_global.repeat(rs, 1, 1)

        # apply local assimilation engine and project onto global latent vectors
        tokens_global_unmasked, posteriors = self.assimilate_local_project_chunked(
            tokens,
            tokens_global,
            cell_lens,
            model_params.q_cells_lens,
            batch.tokens_lens if enable_similarity_diag else None,
        )

        # apply aggregation engine on unmasked tokens
        tokens_global_unmasked = self.aggregation_engine_unmasked(
            tokens_global_unmasked,
            tokens_global_register_class,
            batch.tokens_lens,
            rope_cell_coords=model_params.rope_cell_coords,
        )

        # final processing
        # tokens_global: (rs * num_healpix_cells, num_queries, global_dim)
        # reshape to (rs, num_healpix_cells, num_queries * global_dim) — C-contiguous,
        # so cell j gets [q0_j, q1_j, ...] which is the correct ordering for the mask fill
        # and the subsequent reshape+flatten into the global token sequence.
        tokens_global = tokens_global.reshape(rs, self.num_healpix_cells, -1)
        # prepend register/class tokens per batch sample (no-op when num_extra_tokens == 0)
        if num_extra_tokens > 0:
            tokens_global = torch.cat([tokens_global_register_class, tokens_global], dim=1)

        # create mask from cell lens
        mask_reg_class_tokens = (
            torch.ones(
                self.num_register_tokens + self.num_class_tokens,
                device=tokens_global.device,
            )
            .to(torch.bool)
            .unsqueeze(0)
            .repeat(rs, 1)
        )
        cell_lens_r = cell_lens.unsqueeze(0).reshape(rs, self.num_healpix_cells)
        mask = torch.cat([mask_reg_class_tokens, cell_lens_r.to(torch.bool)], dim=1)

        # fill empty tensor using mask for positions of unmasked tokens
        tokens_global[mask] = tokens_global_unmasked.to(tokens_global.dtype)

        # recover batch dimension and build global token list
        num_tokens_tot = self.num_healpix_cells + self.num_register_tokens + self.num_class_tokens
        q_c_shape = self.q_cells.shape
        tokens_global = (
            tokens_global.reshape([rs, num_tokens_tot, q_c_shape[-2], q_c_shape[-1]])
            #  removing this line because else they get added twice? + model_params.pe_global
        ).flatten(1, 2)

        return tokens_global, posteriors
