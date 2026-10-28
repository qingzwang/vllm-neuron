# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENTAL block-sparse decode selection (model.sparse_keep_blocks), on CPU."""

import torch
from hy_v3_neuron.model import sparse_keep_blocks

BS, D = 32, 16


def _setup(n_hist, nb=12, nh=2, kvh=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    k_cache = torch.randn(40, kvh, BS, D, generator=g, dtype=torch.bfloat16) * 0.1
    block_table = torch.randperm(40, generator=g)[: 2 * nb].view(2, nb).to(torch.int32)
    q = torch.randn(nh, 2, D, generator=g)
    return q, k_cache, block_table, torch.tensor(n_hist)


def _hist_tokens(k_cache, table_row, n_tokens):
    """History K of one request as the decode kernel reads it: the table's blocks in order."""
    rows = k_cache[table_row.long()][:, 0].reshape(-1, D)
    return rows[:n_tokens]


def test_full_fraction_is_dense():
    q, kc, bt, n = _setup([300, 37])
    compact, prior = sparse_keep_blocks(q, kc, bt, n, BS, 1.0, 1, 2)
    assert prior.tolist() == [300, 37]
    for b, nn in enumerate(n.tolist()):
        nbv = -(-nn // BS)
        assert compact[b, :nbv].tolist() == bt[b, :nbv].tolist()


def test_keeps_the_block_with_the_matching_key():
    q, kc, bt, n = _setup([360, 360])
    hot = int(bt[0, 5])
    kc[hot, 0, 7] = (q[:, 0].mean(0) * 50).to(torch.bfloat16)  # a key aligned with request 0's query
    # 0.5 of 12 blocks = 6: the sink, the 2 recent ones and 3 chosen by score.
    compact, prior = sparse_keep_blocks(q, kc, bt, n, BS, 0.5, 1, 2)
    nbv = -(-360 // BS)  # 12 blocks
    cnt = (prior[0] - (360 - BS * (nbv - 1))) // BS + 1
    kept = compact[0, :cnt].tolist()
    assert hot in kept
    assert kept[0] == int(bt[0, 0])  # sink first
    assert kept[-2:] == bt[0, nbv - 2 : nbv].tolist()  # recent blocks last, in order
    assert kept == [x for x in bt[0, :nbv].tolist() if x in kept]  # original order kept


def test_compacted_attention_equals_attention_over_kept_blocks():
    q, kc, bt, n = _setup([333, 333], seed=3)
    compact, prior = sparse_keep_blocks(q, kc, bt, n, BS, 0.4, 1, 2)
    nbv = -(-333 // BS)
    fill = 333 - BS * (nbv - 1)
    cnt = int((prior[0] - fill) // BS + 1)
    kept = compact[0, :cnt].tolist()
    # Reference: the kept blocks' tokens taken from the original table (full
    # blocks, the last one partial).
    ref = []
    for i, blk in enumerate(bt[0, :nbv].tolist()):
        if blk in kept:
            ref.append(kc[blk, 0, : (fill if i == nbv - 1 else BS)])
    ref = torch.cat(ref)
    got = _hist_tokens(kc, compact[0], int(prior[0]))
    assert torch.equal(got, ref)
