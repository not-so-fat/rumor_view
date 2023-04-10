import pytest
import numpy
from numpy import testing as np_testing
import pandas
from pandas import testing as pd_testing
from rumor_view import RumorView


@pytest.fixture(scope="session")
def view_sample():
    return RumorView(
        pandas.DataFrame({
            "c1": [0, 1, 0, 1, 0, 1],
            "c2": ["A", "A", "A", "A", "A", "A"],
            "c3": ["A", "B", "B", "B", "B", "B"],
            "c4": [2, 0, 1, 2, 2, 0],
            "c5": [5, 4, 3, 2, 1, 0]
        })
    )


def test_node_names(view_sample):
    view = view_sample
    assert view_sample.node_names == [
        "c1-0", "c1-1", "c2-A", "c3-A", "c3-B", "c4-0", "c4-1", "c4-2"
    ] + [f"c5-{i}" for i in range(6)]


def test_freq_matrix(view_sample):
    view = view_sample
    np_testing.assert_array_almost_equal(
        view.freq_matrix.toarray(),
        numpy.array([
            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
        ])
    )


def test_co_matrix(view_sample):
    view = view_sample
    np_testing.assert_array_almost_equal(
        view.co_matrix,
        numpy.array([
            [3, 0, 3, 1, 2, 0, 1, 2, 0, 1, 0, 1, 0, 1],
            [0, 3, 3, 0, 3, 2, 0, 1, 1, 0, 1, 0, 1, 0],
            [3, 3, 6, 1, 5, 2, 1, 3, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [2, 3, 5, 0, 5, 2, 1, 2, 1, 1, 1, 1, 1, 0],
            [0, 2, 2, 0, 2, 2, 0, 0, 1, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [2, 1, 3, 1, 2, 0, 0, 3, 0, 1, 1, 0, 0, 1],
            [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
        ])
    )


def test_node_prob(view_sample):
    view = view_sample
    np_testing.assert_array_almost_equal(
        view.node_prob,
        numpy.array([0.5, 0.5, 1.0, 1/6, 5/6, 1/3, 1/6, 1/2, 1/6, 1/6, 1/6, 1/6, 1/6, 1/6])
    )


def test_cprob(view_sample):
    view = view_sample
    pd_testing.assert_frame_equal(
        view._extract_matrix_for_2columns(view.cprob_df, "c4", "c1"),
        pandas.DataFrame(
            numpy.array([
                [0, 2/3],
                [1/3, 0],
                [2/3, 1/3]
            ]),
            columns=["c1-0", "c1-1"],
            index=["c4-0", "c4-1", "c4-2"]
        )
    )


def test_lift(view_sample):
    view = view_sample
    pd_testing.assert_frame_equal(
        view._extract_matrix_for_2columns(view.lift_df, "c4", "c1"),
        pandas.DataFrame(
            numpy.array([
                [0, 2],
                [2, 0],
                [4/3, 2/3]
            ]),
            columns=["c1-0", "c1-1"],
            index=["c4-0", "c4-1", "c4-2"]
        )
    )
