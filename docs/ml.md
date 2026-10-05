# The local models

Every content decision in onnx-web-filter is made by small ONNX models that
run inside the filter's own process with
[ONNX Runtime](https://github.com/microsoft/onnxruntime): whether an image
is explicit, whether a page's text is adult, and what kind of site a host
is (shopping, news, social media, banking, …). Nothing is sent to any
external service, and there is no model server to run.

## Runtime

`webfilter` downloads Microsoft's prebuilt ONNX Runtime release for the
machine (pinned to 1.30.0 in `internal/ml/ortrt/platform.go`, every archive
checked against its published SHA-256), unpacks it under
`data/ml/runtime/`, and loads the shared library through the official Go
bindings (`github.com/microsoft/onnxruntime/go`). The library is loaded
once per process; *Reload models* on the Models page reopens the model
sessions but not the library.

| Setting `ml.accel` | Build |
|---|---|
| `auto` (default) | NVIDIA driver present (x64 Linux/Windows) → CUDA 12 build; otherwise CPU |
| `cpu` | CPU build |
| `cuda` / `cuda13` | CUDA 12 / CUDA 13 build (CUDA 13 needs driver 580 or newer) |

The CUDA builds are 240–440 MB and need the CUDA and cuDNN 9 libraries on
the machine. If the GPU build cannot be downloaded the CPU build is used;
if a model cannot start on the GPU it is opened on the CPU and the Models
page says why (`cpu (cuda failed: …)`). macOS uses the CPU build (Apple
silicon only; there is no macOS x64 build of 1.30).

To use an ONNX Runtime you installed yourself, set `ml.ort_lib_path` (the
library or its directory) or the `ORT_LIB_PATH` environment variable.
Any release from 1.17 on loads.

## Models

| id | Task | Size | Licence | Notes |
|---|---|---|---|---|
| `nsfwjs-mobilenet` | images | 17 MB | MIT | Default. GantMan/NSFWJS MobileNetV2, int8; drawings / hentai / neutral / porn / sexy. Score = porn + hentai + ½ sexy. |
| `falconsai-vit` | images | 83 MB | Apache-2.0 | Falconsai ViT-base, int8; normal / nsfw. Better on photos, about 4× slower. |
| `distilbert-nsfw` | page text | 65 MB | Apache-2.0 | Default. DistilBERT NSFW classifier, int8, English; first 256 tokens of title + text. |
| `minilm-l6` | site categories | 22 MB | Apache-2.0 | Default. all-MiniLM-L6-v2 sentence embeddings, int8. |

Every file is pinned in `internal/ml/catalog/models.json` to a repository
commit, a size and a SHA-256; a download that does not match is rejected,
and an install from an older pin is fetched again. Files come from Hugging
Face (`HF_ENDPOINT` and `HF_TOKEN` are honoured). Pick the model for each
task on the Settings page (or `ml.image_model`, `ml.text_model`,
`ml.site_model`), then download and reload from the Models page.

### Site categories without training

The site model was never trained on the 32 categories. Each category is
described by its label and description plus a few example sites
(`internal/sitecat/examples.go`); at load time these are embedded once.
A site is embedded as `host title description` and compared with every
example; a category scores its best match, and a softmax over those scores
gives the confidence. On 100 held-out sites
(`internal/ml/classify/testdata/sites.tsv`) this gets 84% right with page
titles and 98% within the top three. From the hostname alone only 43% are
right, but those answers usually come with low confidence, and the filter
asks again once the page title is known (see below); hostname-only answers
it is sure of (confidence ≥ 0.6) are right 81% of the time. Better example
sites in `examples.go` are the cheapest way to improve it.

## How requests are classified

```
request ──► decision cache (SQLite + in-memory LRU)  ──hit──► act
               │ miss
               ▼
            prefilters: tiny images, very short pages ─────► allow
               │
               ▼
            job queue (deduplicated by content hash, prioritised)
               │
               ▼  wait at most the budget
            ONNX model  ──verdict──► cache ──► act
                                        │
             budget elapsed ────────────┴──► policy on_timeout action; verdict still cached
```

* **Scores and thresholds.** Each model gives an adult score from 0 to 1.
  A policy blocks (or blurs) at its `threshold` (0.8 for text, 0.75 for
  images and video stills by default). A score of `ml.adult_score` (0.9)
  or more counts as certainly adult: it is blocked whatever the threshold,
  and it counts towards learning the whole site (below).
* **Images** are keyed by exact hash and by a perceptual hash, so the same
  picture at another size or encoding is answered from cache. They are
  downscaled to `ml.max_image_px` (384 px) for the cache and resized to the
  model's 224 px input.
* **Pages** are keyed by a hash of the extracted text (title, description,
  headings, first 3 KB of visible text); the model reads the first 256
  tokens.
* **Sites** are learned: after three certainly-adult verdicts on one
  registrable domain, the whole site is treated as adult.
* **Site categories** are keyed by registrable domain. Installed domain
  lists answer first; the model is asked only about sites they do not know,
  from the hostname on the first navigation and again with the page title
  when that answer was unsure (confidence below 0.6). Only navigations wait
  for it (`ml.budget.category_ms`).
* **Ad and tracker hosts** are judged by EasyList/EasyPrivacy only; no model
  is asked. Hosts can still be blocked by hand on the Decisions page.
* **Prefetch**: when a page passes, the images it references are scored in
  the background so the browser's image requests hit a warm cache.
* **Budgets** (`ml.budget`, milliseconds; per-policy `budget_ms` overrides)
  bound how long a request waits: 500 ms each for images, pages and
  categories. On timeout the policy's `on_timeout` applies; the model
  finishes anyway and the verdict lands in the cache.
* **Unavailable** (models not loaded, queue full): the policy's
  `on_unavailable` applies, `allow` by default.

All of this is visible on the **Decisions** page, where any verdict can be
overridden, and in `/metrics` (`webfilter_ml_*`, `webfilter_verdict_*`).

## Speed

On 3 cores of a 2.1 GHz Xeon (no GPU), new content each time, including
decoding and tokenizing: an image takes about 40 ms with `nsfwjs-mobilenet`
(150 ms with `falconsai-vit`), a page about 80 ms, a site category about
6 ms. Loading all three models takes about 2 s, most of it embedding the
category examples. A cache hit takes microseconds.

| Setting | Effect |
|---|---|
| `ml.parallel` | verdicts run at once; 0 picks 2 on a CPU and 4 with CUDA. |
| `ml.threads` | threads per model session; 0 divides the cores by `ml.parallel`. |
| `ml.max_image_px` | longest side images are reduced to before caching. |

## Command line

```
webfilter ml status            runtime, models and providers
webfilter ml models            the catalog
webfilter ml download [ID]     fetch the runtime and the configured (or named) model
webfilter ml remove ID         delete an installed model that is not configured
webfilter ml test image FILE   score an image with the configured model
webfilter ml test text TEXT    score text
webfilter ml test site HOST [TITLE] [DESCRIPTION]
```

## Developing

Tests that need the real runtime and models run when `ML_DATA_DIR` points
at a data directory filled by `webfilter ml download`:

```
ML_DATA_DIR=$PWD/data/ml go test -race ./internal/ml/...
```

They check the Go preprocessing, tokenizer and pooling against reference
outputs from Python onnxruntime (`internal/ml/classify/testdata/gen_fixtures.py`,
`internal/ml/tokenize/testdata/gen_golden.py`) and report site-category
accuracy.
