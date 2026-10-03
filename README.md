# onnx-web-filter

A policy-based, TLS-intercepting web-filtering proxy for a household or small
office, whose content decisions are made by **small local ONNX models**
running in-process with [ONNX Runtime](https://github.com/microsoft/onnxruntime).
Nothing leaves your network, and there is no model server to run.

It is a fork of [llama-web-filter](https://github.com/Yjlion/llama-web-filter)
(itself a fork of [gowebfilter](https://github.com/Yjlion/gowebfilter)) that
replaces the llama.cpp language model with purpose-built classifiers. A
verdict takes tens of milliseconds on a CPU instead of seconds, and the
first-run download is about 120 MB instead of 3 GB.

- **Adult images, text and video**: an image classifier (NSFWJS MobileNetV2
  by default, or a ViT) blurs, blanks or blocks pictures and video stills; a
  DistilBERT classifier blocks adult pages. Every verdict is cached by
  content, so a picture is only ever judged once; near-duplicates and whole
  sites are learned.
- **Site categories** — shopping, news, social media, banking, games and 27
  more; block some per policy, or allow only some. Installed domain lists
  answer first; for the rest a sentence-embedding model compares the site's
  name and title with examples of each category.
- **Ad and tracker removal** with EasyList/EasyPrivacy (snapshot built in,
  updatable) and cosmetic hiding.
- **Real-time** operation: a decision cache, a deduplicating job queue, per
  request wait budgets with policy-chosen fallbacks and speculative
  pre-scoring of a page's images.
- **Pinned, verified downloads**: Microsoft's prebuilt ONNX Runtime 1.30
  (CPU, or CUDA when an NVIDIA GPU is present) and every model file are
  checked against pinned SHA-256 digests.

Everything gowebfilter had is still there: per-client policies (MAC, IP,
CIDR), URL and category lists, SafeSearch, YouTube channel filtering, DoH and
QUIC blocking, MITM control, SOCKS and transparent listeners, ICAP for Squid,
a management UI with logs and analytics, PAC distribution, Prometheus metrics.

## Quick start

```sh
./webfilter setup        # creates config, downloads ONNX Runtime + models (~120 MB)
./webfilter run          # proxy on :8080 / :1080, management UI on http://127.0.0.1:8000
```

Install the CA certificate from **Settings → Certificates** on your devices,
point them at the proxy (or use `/proxy.pac`), then switch on the
classifiers and categories you want on the **Policies** page. Full steps in
[docs/install.md](docs/install.md).

## Documentation

- [Installing](docs/install.md) — download, setup, running as a service, where files live
- [The local models](docs/ml.md) — runtime, models, how classification and caching work, speed
- [Policies and site categories](docs/policies.md) — category filtering, classifier and ad-block settings
- [Architecture](docs/architecture.md) — pipeline and packages
- [Docker](docs/docker.md), [ICAP with Squid](docs/icap.md), [Metrics](docs/metrics.md)

## Command line

```
webfilter run            proxy + management UI in one process
webfilter setup          first-run wizard
webfilter ml ...         status | models | download | remove | test image|text|site
webfilter rules ...      list | remove   (sentence rules from llama-web-filter)
webfilter adblock ...    status | update
webfilter categories update
webfilter proxy / mgmt   run the two halves separately
```

## Building and testing

The official ONNX Runtime Go bindings use CGO, so building needs Go 1.26+
and a C compiler (gcc or clang; MinGW gcc on Windows). Releases are built
natively for Linux x64/ARM64, macOS on Apple silicon and Windows x64.

```sh
go build ./... && go vet ./... && go test ./...
```

Plain `go test` downloads nothing. To also test against the real runtime
and models (parity with Python onnxruntime, site-category accuracy):

```sh
./webfilter ml download
ML_DATA_DIR=$PWD/data/ml go test -race ./internal/ml/...
```

## Status

Young. The text model reads English; other languages are judged mostly by
the keyword pre-filter and the image model. Site categories from the
hostname alone are often unsure and get re-checked once the page title is
known. Video classification works from posters, thumbnails and (optionally)
ffmpeg keyframes; HLS/DASH segment streams are judged by their poster only.
llama-web-filter's natural-language policy assistant needs a chat model and
is not included.

## License

MIT. The models keep their own licences (MIT and Apache-2.0; see
[docs/ml.md](docs/ml.md)).
