# Restore crates.io Publishing

Date: 2026-05-02

crates.io publishing is intentionally disabled in `.github/workflows/release.yml`.

Reason: llmctl currently depends on a vendored genai patch via:

```toml
[patch.crates-io]
genai = { path = "vendor/genai" }
```

That patch is required for OpenAI-compatible provider behavior documented in `docs/vendored_genai_patch.md`. A crate published to crates.io cannot rely on this repository-local patch in the same way GitHub Release, Homebrew, or `cargo install --git` can.

## Current Release Policy

Supported:

- GitHub Release binaries
- Homebrew tap formula updates
- `cargo install --git https://github.com/zhangzhenhu/llmctl.git`
- local `cargo install --path .`

Disabled:

- `cargo publish`
- `cargo install llmctl` from crates.io

## How To Restore

Only re-enable crates.io publishing after one of these is true:

1. Upstream genai releases all changes listed in `docs/vendored_genai_patch.md`.
2. llmctl switches to a separate registry-published genai fork package.
3. llmctl removes behavior that depends on the vendored genai patch.

Then:

1. Remove `[patch.crates-io]` from `Cargo.toml`.
2. Update `Cargo.lock`.
3. Replace README `cargo install --git` instructions with `cargo install llmctl`.
4. Change `.github/workflows/release.yml`:
   - restore the `publish-crate` job condition to release tags, for example:

```yaml
if: ${{ needs.release.outputs.rc == 'false' }}
```

   - make `update-homebrew` depend on both jobs again if the Homebrew update should wait for crates.io:

```yaml
needs: [release, publish-crate]
```

5. Add package verification back to the validate job:

```yaml
- name: Verify package
  run: cargo package
```

6. Run local checks:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test -q
./scripts/run-cli-dry-run-tests.sh
cargo package --allow-dirty
```
