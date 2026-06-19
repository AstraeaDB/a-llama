set shell := ["bash", "-euo", "pipefail", "-c"]

default:
    @just --list

# Run the embedded demo end-to-end.
run:
    cargo run --release --quiet

# Run the test suite (empty in the template; real projects add theirs).
test:
    cargo test --release --quiet

# Run benchmarks (none in the template).
bench:
    @echo "(no benchmarks in template)"

# Remove build artifacts.
clean:
    cargo clean
