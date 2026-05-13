"""Tests for ``zarr_vectors.core.paths`` — the multiscale link path helpers."""

from __future__ import annotations

import pytest

from zarr_vectors.core.paths import (
    cross_chunk_link_attributes_path,
    cross_chunk_links_path,
    format_delta,
    link_attributes_path,
    links_path,
    parse_delta,
)


@pytest.mark.parametrize(
    "delta,expected",
    [(0, "0"), (1, "+1"), (-1, "-1"), (2, "+2"), (-3, "-3"), (10, "+10")],
)
def test_format_delta_round_trip(delta, expected):
    assert format_delta(delta) == expected
    assert parse_delta(expected) == delta


@pytest.mark.parametrize("bad", ["", " ", "abc", "01", "+", "-"])
def test_parse_delta_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_delta(bad)


def test_links_path_default_is_delta_zero():
    assert links_path() == "links/0"
    assert links_path(0) == "links/0"
    assert links_path(1) == "links/+1"
    assert links_path(-2) == "links/-2"


def test_cross_chunk_links_path():
    assert cross_chunk_links_path() == "cross_chunk_links/0"
    assert cross_chunk_links_path(3) == "cross_chunk_links/+3"
    assert cross_chunk_links_path(-1) == "cross_chunk_links/-1"


def test_link_attributes_path():
    assert link_attributes_path("weight") == "link_attributes/weight/0"
    assert link_attributes_path("weight", 1) == "link_attributes/weight/+1"
    assert link_attributes_path("kind", -2) == "link_attributes/kind/-2"


def test_cross_chunk_link_attributes_path():
    assert cross_chunk_link_attributes_path("weight") == \
        "cross_chunk_link_attributes/weight/0"
    assert cross_chunk_link_attributes_path("weight", 2) == \
        "cross_chunk_link_attributes/weight/+2"
