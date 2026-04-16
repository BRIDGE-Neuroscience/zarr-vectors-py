"""Command-line interface for zarr-vectors.

Usage::

    zarr-vectors ingest points  scan.las    scan.zarrvectors   --chunk-shape 100,100,50
    zarr-vectors ingest streams tracts.trk  tracts.zarrvectors --chunk-shape 50,50,50
    zarr-vectors ingest skeleton neuron.swc neuron.zarrvectors --chunk-shape 200,200,200
    zarr-vectors ingest mesh    brain.obj   brain.zarrvectors  --chunk-shape 100,100,100

    zarr-vectors export csv  store.zarrvectors output.csv
    zarr-vectors export obj  store.zarrvectors output.obj
    zarr-vectors export swc  store.zarrvectors output.swc

    zarr-vectors build-pyramid store.zarrvectors --reduction-factor 8

    zarr-vectors validate store.zarrvectors --level 3

    zarr-vectors info store.zarrvectors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_chunk_shape(s: str) -> tuple[float, ...]:
    """Parse comma-separated chunk shape string."""
    return tuple(float(x.strip()) for x in s.split(","))


# ===================================================================
# Ingest subcommands
# ===================================================================

def _cmd_ingest_points(args: argparse.Namespace) -> None:
    ext = Path(args.input).suffix.lower()
    cs = _parse_chunk_shape(args.chunk_shape)

    if ext in (".las", ".laz"):
        from zarr_vectors.ingest.las import ingest_las
        result = ingest_las(args.input, args.output, cs, dtype=args.dtype)
    elif ext in (".ply",):
        from zarr_vectors.ingest.ply import ingest_ply
        result = ingest_ply(args.input, args.output, cs, dtype=args.dtype)
    elif ext in (".csv", ".xyz", ".txt"):
        from zarr_vectors.ingest.csv_points import ingest_csv
        delimiter = " " if ext == ".xyz" else ","
        has_header = ext != ".xyz"
        result = ingest_csv(
            args.input, args.output, cs,
            delimiter=delimiter, has_header=has_header, dtype=args.dtype,
        )
    else:
        print(f"Unknown point format: {ext}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingested {result['vertex_count']} vertices into {args.output}")


def _cmd_ingest_streams(args: argparse.Namespace) -> None:
    ext = Path(args.input).suffix.lower()
    cs = _parse_chunk_shape(args.chunk_shape)

    if ext == ".trk":
        from zarr_vectors.ingest.trk import ingest_trk
        result = ingest_trk(args.input, args.output, cs, dtype=args.dtype)
    elif ext == ".tck":
        from zarr_vectors.ingest.tck import ingest_tck
        result = ingest_tck(args.input, args.output, cs, dtype=args.dtype)
    elif ext == ".trx":
        from zarr_vectors.ingest.trx import ingest_trx
        result = ingest_trx(args.input, args.output, cs, dtype=args.dtype)
    else:
        print(f"Unknown streamline format: {ext}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingested {result['polyline_count']} streamlines "
          f"({result['vertex_count']} vertices) into {args.output}")


def _cmd_ingest_skeleton(args: argparse.Namespace) -> None:
    ext = Path(args.input).suffix.lower()
    cs = _parse_chunk_shape(args.chunk_shape)

    if ext == ".swc":
        from zarr_vectors.ingest.swc import ingest_swc
        result = ingest_swc(args.input, args.output, cs, dtype=args.dtype)
    elif ext in (".graphml", ".xml"):
        from zarr_vectors.ingest.graphml import ingest_graphml
        result = ingest_graphml(args.input, args.output, cs, dtype=args.dtype)
    else:
        print(f"Unknown skeleton format: {ext}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingested {result['node_count']} nodes into {args.output}")


def _cmd_ingest_mesh(args: argparse.Namespace) -> None:
    ext = Path(args.input).suffix.lower()
    cs = _parse_chunk_shape(args.chunk_shape)
    encoding = args.encoding or "raw"

    if ext == ".obj":
        from zarr_vectors.ingest.obj import ingest_obj
        result = ingest_obj(
            args.input, args.output, cs,
            dtype=args.dtype, encoding=encoding,
        )
    elif ext == ".stl":
        from zarr_vectors.ingest.stl import ingest_stl
        result = ingest_stl(
            args.input, args.output, cs,
            dtype=args.dtype, encoding=encoding,
        )
    else:
        print(f"Unknown mesh format: {ext}", file=sys.stderr)
        sys.exit(1)

    print(f"Ingested {result['vertex_count']} vertices, "
          f"{result['face_count']} faces into {args.output}")


# ===================================================================
# Export subcommands
# ===================================================================

def _cmd_export(args: argparse.Namespace) -> None:
    fmt = args.format.lower()

    if fmt == "csv":
        from zarr_vectors.export.csv_points import export_csv
        result = export_csv(args.store, args.output)
        print(f"Exported {result['vertex_count']} vertices to {args.output}")
    elif fmt == "obj":
        from zarr_vectors.export.obj import export_obj
        result = export_obj(args.store, args.output)
        print(f"Exported {result['vertex_count']} vertices, "
              f"{result['face_count']} faces to {args.output}")
    elif fmt == "swc":
        from zarr_vectors.export.swc import export_swc
        result = export_swc(args.store, args.output)
        print(f"Exported {result['node_count']} nodes to {args.output}")
    elif fmt == "ply":
        from zarr_vectors.export.ply import export_ply
        result = export_ply(args.store, args.output)
        print(f"Exported {result['vertex_count']} vertices to {args.output}")
    elif fmt == "trk":
        from zarr_vectors.export.trk import export_trk
        result = export_trk(args.store, args.output)
        print(f"Exported {result['streamline_count']} streamlines to {args.output}")
    elif fmt == "trx":
        from zarr_vectors.export.trx import export_trx
        result = export_trx(args.store, args.output)
        print(f"Exported {result['streamline_count']} streamlines to {args.output}")
    else:
        print(f"Unknown export format: {fmt}", file=sys.stderr)
        sys.exit(1)


# ===================================================================
# Pyramid
# ===================================================================

def _cmd_build_pyramid(args: argparse.Namespace) -> None:
    from zarr_vectors.multiresolution.coarsen import build_pyramid

    result = build_pyramid(
        args.store,
        reduction_factor=args.reduction_factor,
    )
    print(f"Created {result['levels_created']} resolution level(s)")
    for spec in result.get("level_specs", []):
        print(f"  level {spec['level']}: bin_size={spec['bin_size']:.1f}, "
              f"~{spec['expected_vertices']} vertices")


# ===================================================================
# Validate
# ===================================================================

def _cmd_validate(args: argparse.Namespace) -> None:
    from zarr_vectors.validate import validate

    result = validate(args.store, level=args.level)
    print(result.summary())
    sys.exit(0 if result.ok else 1)


# ===================================================================
# Info
# ===================================================================

def _cmd_info(args: argparse.Namespace) -> None:
    from zarr_vectors.core.store import (
        open_store, read_root_metadata, list_resolution_levels,
        get_resolution_level, store_info,
    )
    from zarr_vectors.core.arrays import list_chunk_keys

    try:
        root = open_store(str(args.store))
        meta = read_root_metadata(root)
    except Exception as e:
        print(f"Cannot open store: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Store: {args.store}")
    print(f"  SID dimensions: {meta.sid_ndim}D")
    print(f"  Chunk shape: {meta.chunk_shape}")
    print(f"  Bounds: {meta.bounds}")
    print(f"  Geometry types: {meta.geometry_types}")
    print(f"  Links convention: {meta.links_convention}")
    print(f"  Object index: {meta.object_index_convention}")
    print(f"  Cross-chunk: {meta.cross_chunk_strategy}")

    levels = list_resolution_levels(root)
    print(f"  Resolution levels: {len(levels)}")
    for li in levels:
        try:
            lg = get_resolution_level(root, li)
            attrs = lg.attrs
            vc = attrs.get("vertex_count", "?")
            bs = attrs.get("bin_shape", attrs.get("bin_size", None))
            bs_str = f", bin_shape={bs}" if bs else ""
            print(f"    resolution_{li}: {vc} vertices{bs_str}")
        except Exception:
            print(f"    resolution_{li}: (unreadable)")

    try:
        info = store_info(root)
        if "total_bytes" in info:
            mb = info["total_bytes"] / (1024 * 1024)
            print(f"  Total size: {mb:.2f} MB")
    except Exception:
        pass


# ===================================================================
# Main parser
# ===================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zarr-vectors",
        description="Tools for zarr vectors data",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- ingest ---
    ingest_parser = sub.add_parser("ingest", help="Ingest data into a zarr vectors store")
    ingest_sub = ingest_parser.add_subparsers(dest="ingest_type")

    for name, func, help_text in [
        ("points", _cmd_ingest_points, "Point clouds (LAS, PLY, CSV, XYZ)"),
        ("streams", _cmd_ingest_streams, "Streamlines (TRK, TCK, TRX)"),
        ("skeleton", _cmd_ingest_skeleton, "Skeletons (SWC, GraphML)"),
        ("mesh", _cmd_ingest_mesh, "Meshes (OBJ, STL)"),
    ]:
        p = ingest_sub.add_parser(name, help=help_text)
        p.add_argument("input", help="Input file path")
        p.add_argument("output", help="Output zarr vectors store path")
        p.add_argument("--chunk-shape", required=True, help="Comma-separated chunk dimensions")
        p.add_argument("--dtype", default="float32", help="Position dtype (default: float32)")
        if name == "mesh":
            p.add_argument("--encoding", default="raw", choices=["raw", "draco"])
        p.set_defaults(func=func)

    # --- export ---
    export_parser = sub.add_parser("export", help="Export from a zarr vectors store")
    export_parser.add_argument("format", choices=["csv", "obj", "swc", "ply", "trk", "trx"])
    export_parser.add_argument("store", help="Zarr vectors store path")
    export_parser.add_argument("output", help="Output file path")
    export_parser.set_defaults(func=_cmd_export)

    # --- build-pyramid ---
    pyr_parser = sub.add_parser("build-pyramid", help="Build multi-resolution pyramid")
    pyr_parser.add_argument("store", help="Zarr vectors store path")
    pyr_parser.add_argument("--reduction-factor", type=int, default=8,
                             help="Min fold-reduction per level (default: 8)")
    pyr_parser.set_defaults(func=_cmd_build_pyramid)

    # --- validate ---
    val_parser = sub.add_parser("validate", help="Validate a store")
    val_parser.add_argument("store", help="Zarr vectors store path")
    val_parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3, 4, 5],
                             help="Conformance level (default: 3)")
    val_parser.set_defaults(func=_cmd_validate)

    # --- info ---
    info_parser = sub.add_parser("info", help="Show store information")
    info_parser.add_argument("store", help="Zarr vectors store path")
    info_parser.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
