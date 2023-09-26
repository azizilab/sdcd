import pytest

from ..models import (
    SDCI,
    DCDI,
    DCDFG,
    DAGMA,
    NOBEARS,
    NOTEARS,
    Sortnregress,
)

from .utils import generate_test_dataset


@pytest.fixture
def interventional_dataset():
    return generate_test_dataset(n=10, d=5)


@pytest.fixture
def observational_dataset():
    return generate_test_dataset(n=10, d=5, use_interventions=False)


def test_sdci(interventional_dataset):
    m = SDCI()
    m.train(interventional_dataset, finetune=True)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_sdci_gumbel(interventional_dataset):
    m = SDCI(use_gumbel=True)
    m.train(interventional_dataset, finetune=True)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)
    m.fix_gumbel_threshold()
    assert not m._model.layers[0].gumbel_adjacency.log_alpha.requires_grad


def test_dcdi(interventional_dataset):
    m = DCDI()
    m.train(interventional_dataset, max_epochs=10, finetune=True)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_dcdfg(interventional_dataset):
    m = DCDFG()
    m.train(interventional_dataset, num_modules=5, max_epochs=2, finetune=True)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_dagma(observational_dataset):
    m = DAGMA()
    m.train(observational_dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_notears(observational_dataset):
    m = NOTEARS()
    m.train(observational_dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_nobears(observational_dataset):
    m = NOBEARS()
    m.train(observational_dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_sortnregress(observational_dataset):
    m = Sortnregress()
    m.train(observational_dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)
