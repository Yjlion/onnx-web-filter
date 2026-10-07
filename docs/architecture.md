# Architecture

```
client ──► listeners (HTTP proxy, SOCKS5, transparent, ICAP)
              │   TLS is terminated with a locally issued leaf certificate
              ▼
   addon pipeline  (internal/app/engine.go, order is load-bearing)
   ManagementAccess → ProxyAuthGate → PolicyRouter → RuleEvaluator → MitmControl
   → UrlFilter → CategoryFilter → AdBlocker → QuicBlocker → DohFilter → SafeSearch → YouTubeFilter
   → TextClassifier → ImageClassifier → RequestLogger
              │
              ▼
   internal/classify/verdict        decision cache, job queue, budgets, site learning
              │
              ▼
   internal/ml                      ONNX Runtime (in-process), model catalog, classifiers
```

## Packages

| Package | Role |
|---|---|
| `cmd/webfilter` | CLI: `run`, `setup`, `proxy`, `mgmt`, `ml`, `rules`, `adblock`, `categories`, `service` |
| `internal/proxy` | listeners, MITM, upstream fetching, the `FlowContext` every addon sees |
| `internal/proxy/addons` | one file per pipeline stage |
| `internal/proxy/state` | live settings, policies, rules and filter lists with hot reload |
| `internal/models` | settings and policy schema |
| `internal/policy/rules` | sentence rules saved by earlier versions (still enforced) and named devices |
| `internal/sitecat` | the website category taxonomy and the domain-list mapping onto it |
| `internal/classify/verdict` | the decision layer in front of the models |
| `internal/classify/{imageprep,phash,textextract}` | image downscaling, perceptual hashing, HTML text extraction |
| `internal/ml` | the service that loads, reloads and reports on the runtime and models |
| `internal/ml/ortrt` | ONNX Runtime release download and checksum verification (pure Go) |
| `internal/ml/ortenv` | locating and loading the shared library through the official Go bindings (CGO) |
| `internal/ml/catalog` | pinned model catalog and Hugging Face downloads |
| `internal/ml/tokenize` | tokenizer.json WordPiece tokenizer matching the Rust `tokenizers` library |
| `internal/ml/classify` | image and text classifiers, sentence embedder, zero-shot site categories |
| `internal/adblock` | EasyList parser, matcher and cosmetic filtering |
| `internal/mgmtapi` + `ui/` | management REST API and the embedded web UI |

## Data flow for one page

1. `PolicyRouter` matches the client to a policy; `RuleEvaluator` overlays
   the rules that apply now and replaces `fc.Policy` with the effective one.
2. `UrlFilter`, `CategoryFilter` and `AdBlocker` decide on the URL alone
   (lists, cache, and for a navigation the model's category within a
   500 ms budget; ad hosts come from EasyList alone).
3. The response is fetched identity-encoded and buffered.
4. `CategoryFilter` categorizes a navigation's page from its own content
   (cached per URL and content hash, so a changed page is judged again) and
   blocks it if the page's category is refused. `AdBlocker` injects cosmetic CSS; `TextClassifier` extracts the page's
   text, asks the verdict service (cache → model within the budget) and
   blocks or passes; it also hands the page's image URLs to the prefetcher.
5. `ImageClassifier` does the same per image (and per inline data URI),
   replacing adult images with a blurred, checkerboard or blank stand-in.
6. `RequestLogger` records the final action.

## Why the models are behind a cache and a queue

The models answer in tens of milliseconds on a CPU, but web pages reference
dozens of images and a busy household loads many pages at once. Every
verdict is
cached by content (exact hash, perceptual hash for images, text hash for
pages, URL plus content hash for page categories, registrable domain for
learned sites and site categories), so anything seen once is free
forever; and the proxy never waits longer than a budget, applying a
policy-chosen fallback instead while the model finishes in the background.
Requests for the same content share one model call, and a bounded queue
refuses work rather than building a backlog nobody will wait for.

## Hot reload

`policies/*.json`, `config/settings.json` (the hot subset) and
`config/rules.json` are watched; filter lists are reloaded after an update.
The models are only reloaded on request (Models page), because loading
them and embedding the category examples takes about two seconds. The ONNX
Runtime library itself stays loaded for the life of the process.

## Build

The official ONNX Runtime Go bindings use CGO, so `webfilter` is built
natively on each platform (Linux x64/arm64, macOS arm64, Windows x64 with
MinGW gcc) rather than cross-compiled. The runtime library is not linked:
it is loaded with `dlopen`/`LoadLibrary` at start-up, so one binary works
with the CPU and the CUDA builds.
