# SPDX-License-Identifier: Apache-2.0
"""EXPERIMENTAL block-sparse decode (HY3_SPARSE_DECODE) helpers, on CPU."""

import torch
from hy_v3_neuron.model import (
    _SUMMARY_EMPTY,
    sparse_budget_blocks,
    sparse_select_blocks,
    update_k_summary_prefill,
)

BS, D, KVH, NBLK = 32, 16, 1, 64


def _empty():
    return (torch.full((NBLK, KVH, D), _SUMMARY_EMPTY, dtype=torch.bfloat16),
            torch.full((NBLK, KVH, D), -_SUMMARY_EMPTY, dtype=torch.bfloat16))


def _slots(table_row, positions):
    return torch.tensor([int(table_row[p // BS]) * BS + p % BS for p in positions])


def test_budget_is_a_multiple_of_256_and_capped():
    assert sparse_budget_blocks(2688, 0.125, 1, 4) == 512  # 336 + 5 -> 512
    assert sparse_budget_blocks(1024, 0.1, 1, 4) == 256
    assert sparse_budget_blocks(200, 0.5, 1, 4) == 200  # not smaller: dense


def test_prefill_summaries_match_the_written_keys():
    g = torch.Generator().manual_seed(0)
    table = torch.randperm(NBLK - 1, generator=g)[:10] + 1  # block 0 is the null block here
    keys = {}
    kmin, kmax = _empty()
    # A stale summary on a reused block must not survive a fresh write.
    kmin[table[2]] = -100.0
    kmax[table[2]] = 100.0

    def prefill(start, n, T):
        k = torch.randn(KVH, T, D, generator=g)
        pos = list(range(start, start + n))
        for i, p in enumerate(pos):  # a re-computed token has the same K
            if p in keys:
                k[:, i] = keys[p]
        slots = torch.full((T,), -1)
        slots[:n] = _slots(table, pos)
        for i, p in enumerate(pos):
            keys[p] = k[:, i]
        update_k_summary_prefill(kmin, kmax, k, slots, BS)

    prefill(0, 64, 64)       # two full blocks
    prefill(64, 41, 64)      # 41 tokens: block 2 full, block 3 partial, padding after
    prefill(104, 1, 32)      # the re-computed last token: continues block 3
    for b in range(4):
        ks = torch.stack([keys[t] for t in range(b * BS, min(105, (b + 1) * BS))])
        torch.testing.assert_close(kmin[table[b]].float(), ks.amin(0).to(torch.bfloat16).float())
        torch.testing.assert_close(kmax[table[b]].float(), ks.amax(0).to(torch.bfloat16).float())


def _keys_and_summaries(n, g):
    table = (torch.randperm(NBLK - 1, generator=g)[:8] + 1).view(1, 8)
    k = torch.randn(KVH, 8 * BS, D, generator=g)
    kmin, kmax = _empty()
    slots = torch.full((8 * BS,), -1)
    slots[:n] = _slots(table[0], range(n))
    update_k_summary_prefill(kmin, kmax, k, slots, BS)
    return table, k, kmin, kmax


def test_bound_is_an_upper_bound_and_the_hot_block_is_kept():
    g = torch.Generator().manual_seed(1)
    n = 250
    table, k, kmin, kmax = _keys_and_summaries(n, g)
    q = torch.randn(2, 1, 1, D, generator=g)  # 2 query heads on the KV head
    k[0, 3 * BS + 5] = q[:, 0, 0].mean(0) * 20  # a key aligned with the query
    kmin, kmax = _empty()
    slots = torch.full((8 * BS,), -1)
    slots[:n] = _slots(table[0], range(n))
    update_k_summary_prefill(kmin, kmax, k, slots, BS)
    # The bound of each block is >= the true max of q . k over its tokens.
    smin, smax = kmin[table[0].long()].float(), kmax[table[0].long()].float()
    for b in range(-(-n // BS)):
        toks = k[0, b * BS : min(n, (b + 1) * BS)]
        true = (q[:, 0, 0] @ toks.T).max()
        bound = (q[:, 0, 0].clamp(min=0) @ smax[b, 0] + q[:, 0, 0].clamp(max=0) @ smin[b, 0]).max()
        assert bound >= true - 1e-2
    compact, prior = sparse_select_blocks(q, kmin, kmax, table, torch.tensor([n]), BS, 5, 1, 2)
    nbv = -(-n // BS)
    kept = compact[0, :4].tolist()  # n_sel - 1 kept, then the block after the history
    assert int(table[0, 3]) in kept
    assert kept[0] == int(table[0, 0]) and kept[-2:] == table[0, nbv - 2 : nbv].tolist()
    assert kept == [x for x in table[0].tolist() if x in kept]  # original order
    assert int(prior[0]) == 3 * BS + (n - BS * (nbv - 1))
    # Past the kept blocks: the block after the history (here the history fills
    # the table, so the index is clamped to its last block).
    assert compact[0, 4:].tolist() == [int(table[0, min(nbv, 7)])]


def test_budget_covering_the_history_is_dense():
    g = torch.Generator().manual_seed(2)
    n = 150
    table, k, kmin, kmax = _keys_and_summaries(n, g)
    q = torch.randn(1, 1, 4, D, generator=g)  # one request, 4 verify tokens
    compact, prior = sparse_select_blocks(q, kmin, kmax, table, torch.tensor([n]), BS, 6, 1, 2)
    nbv = -(-n // BS)
    assert compact[0].tolist() == table[0, :6].tolist()  # = the table, incl. the next block
    assert int(prior[0]) == n


def test_blocks_written_by_decode_are_kept():
    """Blocks past the prompt have no summary (decode does not write one): always kept."""
    g = torch.Generator().manual_seed(3)
    table = (torch.randperm(NBLK - 1, generator=g)[:12] + 1).view(1, 12)
    kmin, kmax = _empty()
    k = torch.randn(KVH, 6 * BS, D, generator=g)
    slots = torch.full((6 * BS,), -1)
    slots[: 5 * BS + 7] = _slots(table[0], range(5 * BS + 7))  # prompt: 5 blocks + 7 tokens
    update_k_summary_prefill(kmin, kmax, k, slots, BS)
    n = 9 * BS + 3  # decode filled the 6th block and wrote into blocks 7-9
    q = torch.randn(1, 1, 1, D, generator=g)
    compact, prior = sparse_select_blocks(q, kmin, kmax, table, torch.tensor([n]), BS, 9, 1, 2)
    kept = compact[0, :8].tolist()
    # Blocks 5 (prompt end, continued by decode) to 9 have to be kept, the sink too.
    for b in [0, 5, 6, 7, 8, 9]:
        assert int(table[0, b]) in kept
    assert int(prior[0]) == 7 * BS + 3



def test_the_block_after_the_history_stays_in_the_table():
    """MTP verify tokens crossing a block boundary are read back through the table."""
    g = torch.Generator().manual_seed(5)
    n = 7 * BS + 30  # 8 history blocks, 30 tokens in the last: a 4-token step crosses into block 8
    table, k, kmin, kmax = _keys_and_summaries(min(n, 8 * BS), g)
    table = torch.cat([table, (torch.arange(4) + 50).view(1, 4)], dim=1)  # blocks 8-11
    q = torch.randn(1, 1, 4, D, generator=g)
    compact, prior = sparse_select_blocks(q, kmin, kmax, table, torch.tensor([n]), BS, 4, 1, 1)
    assert compact.shape == (1, 4)
    assert int(compact[0, 2]) == int(table[0, 7])  # the last history block, kept
    assert int(compact[0, 3]) == int(table[0, 8])  # the next one
    assert int(prior[0]) == 2 * BS + 30
