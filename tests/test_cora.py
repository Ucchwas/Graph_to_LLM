"""Integration: requires the Planetoid download. Verifies the corrected dataset
facts (5,278 undirected edges, not the spec's 5,429) and the split protocol."""
import pytest
import torch

pytest.importorskip("torch_geometric")

from g2l.data import edge_split, load_cora


@pytest.fixture(scope="module")
def cora():
    try:
        return load_cora()
    except Exception as e:
        pytest.skip(f"Cora unavailable: {e}")


def test_cora_facts(cora):
    assert cora.num_nodes == 2708
    assert cora.edge_index.shape == (2, 10556)
    assert cora.x.shape == (2708, 1433)


def test_split_sizes(cora):
    train, val, test = edge_split(cora, seed=0)
    assert train.pos_edge_label_index.shape[1] == 4488
    assert val.pos_edge_label_index.shape[1] == 263
    assert test.pos_edge_label_index.shape[1] == 527
    assert test.neg_edge_label_index.shape[1] == 527


def test_supervision_disjoint_from_message_passing(cora):
    train, _, test = edge_split(cora, seed=0)
    def pairs(ei):
        return set(map(tuple, ei.t().tolist()))
    mp = pairs(test.edge_index)  # train + val edges, both directions
    held_out = pairs(test.pos_edge_label_index)
    assert not (held_out & mp), "held-out test edges visible to message passing"
    rev = {(j, i) for i, j in held_out}
    assert not (rev & mp)
