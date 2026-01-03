<p align="center">
  <img src="./logo.webp" height="300">
</p>

![Crates.io Version](https://img.shields.io/crates/v/fsst_rs)
![docs.rs](https://img.shields.io/docsrs/fsst-rs)
![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/spiraldb/fsst/ci.yml?branch=develop)


# fsst-rs


A pure-Rust, zero-dependency implementation of the [FSST string compression algorithm][whitepaper].

FSST is a string compression algorithm meant for use in database systems. It was designed by
[Peter Boncz, Thomas Neumann, and Viktor Leis][whitepaper]. It provides 1-3GB/sec compression
and decompression of strings at compression rates competitive with or better than LZ4.

This implementation is somewhat inspired by the [MIT-licensed implementation] from the paper authors, written in C++,
but it is mostly written from a careful reading of the paper.

**NOTE: This current implementation is still in-progress and is not production ready, please use at your own risk.**

**NOTE: This crate only works on little-endian architectures currently. There are no current plans to support big-endian targets.**

## Examples

The repository includes several examples demonstrating different use cases:

- `round_trip`: Simple example showing basic compression and decompression
- `file_compressor`: Command-line tool for compressing files
- `email_template_compression`: Demonstrates compression on business documents (email templates)

To run an example:
```bash
cargo run --example email_template_compression
```

[whitepaper]: https://www.vldb.org/pvldb/vol13/p2649-boncz.pdf
[MIT-licensed implementation]: https://github.com/cwida/fsst
