import pytest

from models import (
    SDCI,
    DCDI,
    DCDFG,
    DAGMA,
    NOBEARS,
    NOTEARS,
)

from .utils import generate_test_dataset


@pytest.fixture
def dataset():
    return generate_test_dataset(n=10, d=5)


def test_sdci(dataset):
    m = SDCI()
    m.train(dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_dcdi(dataset):
    m = DCDI()
    m.train(dataset, max_epochs=10)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_dcdfg(dataset):
    m = DCDFG()
    m.train(dataset, num_modules=5, max_epochs=2)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_dagma(dataset):
    m = DAGMA()
    m.train(dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_notears(dataset):
    m = NOTEARS()
    m.train(dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)


def test_nobears(dataset):
    m = NOBEARS()
    m.train(dataset)
    assert m.get_adjacency_matrix(threshold=False).shape == (5, 5)
    assert m.get_adjacency_matrix(threshold=True).shape == (5, 5)
