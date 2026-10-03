package app

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/yjlion/onnx-web-filter/internal/classify/imageprep"
	"github.com/yjlion/onnx-web-filter/internal/classify/verdict"
	"github.com/yjlion/onnx-web-filter/internal/mgmtapi"
	"github.com/yjlion/onnx-web-filter/internal/ml"
	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/classify"
	"github.com/yjlion/onnx-web-filter/internal/models"
	"github.com/yjlion/onnx-web-filter/internal/proxy"
	"github.com/yjlion/onnx-web-filter/internal/proxy/addons"
	"github.com/yjlion/onnx-web-filter/internal/proxy/state"
	"github.com/yjlion/onnx-web-filter/internal/sitecat"
)

// MLStack is everything classification contributes to a running process:
// the service that owns the ONNX models, the verdict service that caches
// and queues decisions in front of it, and the adapters the pipeline and
// the management API talk to.
type MLStack struct {
	Svc      *ml.Service
	Verdicts *verdict.Service
	Store    *verdict.Store
	Prefetch *verdict.Prefetcher
	maxPx    int
}

// NewMLStack builds and starts the stack for settings. It never fails the
// caller: models that cannot load are reported on the Models page and the
// proxy runs with classification unavailable (the configured
// on_unavailable actions apply).
func NewMLStack(ctx context.Context, cfg models.MLConfig) *MLStack {
	svc := ml.New(cfg)
	if !cfg.Enabled {
		slog.Info("ml: disabled in settings; content classification unavailable")
	} else if err := svc.Start(ctx); err != nil {
		slog.Error("ml: models failed to load; content classification unavailable until fixed", "err", err)
	} else if rt, missing := svc.Missing(); rt || len(missing) > 0 {
		slog.Warn("ml: runtime or models not installed; run `webfilter ml download` or use the Models page", "runtime_missing", rt, "models_missing", missing)
	}

	dbPath := ":memory:"
	if cfg.DataDir != "" {
		dbPath = filepath.Join(cfg.DataDir, "decisions.db")
	}
	store, err := verdict.Open(dbPath, 50000)
	if err != nil {
		slog.Error("verdict: cannot open decision cache, using memory", "err", err, "path", dbPath)
		store, _ = verdict.OpenMemory()
	}
	backend := &onnxBackend{svc: svc, adultScore: cfg.AdultScore}
	vs := verdict.New(store, backend, verdict.Options{
		Workers:    svc.Slots(),
		QueueLimit: 64 * svc.Slots(),
		MaxImagePx: cfg.MaxImagePx,
	})
	pre := verdict.NewPrefetcher(vs, &http.Client{Transport: proxy.NewTransport(), Timeout: 20 * time.Second}, 2)
	return &MLStack{Svc: svc, Verdicts: vs, Store: store, Prefetch: pre, maxPx: cfg.MaxImagePx}
}

// Close stops the verdict workers and the models.
func (st *MLStack) Close() {
	st.Verdicts.Close()
	st.Svc.Stop()
	_ = st.Store.Close()
}

// PipelineClassifiers is what BuildProxyEngine wires into the addons.
func (st *MLStack) PipelineClassifiers() Classifiers {
	return Classifiers{Classifier: &pipelineClassifier{vs: st.Verdicts}, Prefetcher: st.Prefetch, Fetcher: st.Prefetch, Sites: st.Verdicts}
}

// Scanner is the management API's content scanner.
func (st *MLStack) Scanner() mgmtapi.ContentScanner { return &scanner{st: st} }

// Controller is the management API's model controller.
func (st *MLStack) Controller() mgmtapi.MLController {
	return &controller{svc: st.Svc, vs: st.Verdicts}
}

// Decisions is the management API's decision-cache view.
func (st *MLStack) Decisions() mgmtapi.DecisionStore { return &decisions{vs: st.Verdicts} }

// ---- verdict.Backend over ml.Service ----

type onnxBackend struct {
	svc        *ml.Service
	adultScore float64
}

func (b *onnxBackend) Ready() bool     { return b.svc.Ready() }
func (b *onnxBackend) ModelID() string { return b.svc.ModelIDs() }
func (b *onnxBackend) Vision() bool    { return true }

// Hosts is false: no model judges ad/tracker hosts, EasyList does.
func (b *onnxBackend) Hosts() bool { return false }

func (b *onnxBackend) Image(ctx context.Context, mime string, data []byte, hint string) (verdict.Result, error) {
	sc, err := b.svc.Image(ctx, data)
	if err != nil {
		return verdict.Result{}, err
	}
	_, top := sc.Top()
	return verdict.Result{Score: sc.Unsafe, Adult: sc.Unsafe >= b.adultScore, Confidence: top, Detail: describe(sc)}, nil
}

func (b *onnxBackend) Text(ctx context.Context, url, title, text string) (verdict.Result, error) {
	in := text
	if t := strings.TrimSpace(title); t != "" {
		in = t + ". " + text
	}
	sc, err := b.svc.Text(ctx, in)
	if err != nil {
		return verdict.Result{}, err
	}
	_, top := sc.Top()
	return verdict.Result{Score: sc.Unsafe, Adult: sc.Unsafe >= b.adultScore, Confidence: top, Detail: describe(sc)}, nil
}

func (b *onnxBackend) Host(ctx context.Context, host string, samplePaths []string) (verdict.Result, error) {
	return verdict.Result{}, errors.New("hosts are not classified by a model")
}

func (b *onnxBackend) Site(ctx context.Context, host, title, description string) (verdict.Result, error) {
	sc, err := b.svc.Site(ctx, host, title, description)
	if err != nil {
		return verdict.Result{}, err
	}
	var parts []string
	for _, r := range sc.Ranked[:min(3, len(sc.Ranked))] {
		parts = append(parts, fmt.Sprintf("%s %.2f", r.Slug, r.Probability))
	}
	return verdict.Result{Category: sc.Category, Confidence: sc.Confidence, Detail: strings.Join(parts, ", ")}, nil
}

// describe lists the most probable labels, e.g. "porn 0.82, sexy 0.10".
func describe(sc classify.Scores) string {
	idx := make([]int, len(sc.Probs))
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return sc.Probs[idx[a]] > sc.Probs[idx[b]] })
	var parts []string
	for _, i := range idx[:min(3, len(idx))] {
		parts = append(parts, fmt.Sprintf("%s %.2f", sc.Labels[i], sc.Probs[i]))
	}
	return strings.Join(parts, ", ")
}

// ---- state.SiteCategorizer over verdict.Service ----

// siteCategorizer answers the pipeline's category questions: the installed
// domain lists first, then the verdict service (cache, then model).
type siteCategorizer struct {
	vs    *verdict.Service
	lists sitecat.ListMatcher
}

// NewSiteCategorizer adapts a verdict service and the domain lists to the
// runtime's categorizer interface. lists may be nil.
func NewSiteCategorizer(vs *verdict.Service, lists sitecat.ListMatcher) state.SiteCategorizer {
	return &siteCategorizer{vs: vs, lists: lists}
}

func (c *siteCategorizer) Categorize(ctx context.Context, q state.CategoryLookup) state.CategoryAnswer {
	a := c.vs.Category(ctx, verdict.CategoryRequest{
		Host: q.Host, ListCategory: sitecat.FromLists(c.lists, sitecat.HostOf(q.Host)),
		Title: q.Title, Description: q.Description, Enqueue: q.Enqueue, Budget: q.Budget,
	})
	return state.CategoryAnswer{Category: a.Category, Source: string(a.Source), Confidence: a.Confidence,
		Known: a.Known && a.Category != "", TimedOut: a.TimedOut, Unavailable: a.Unavailable}
}

// ---- addons.ContentClassifier over verdict.Service ----

type pipelineClassifier struct{ vs *verdict.Service }

func toVerdict(a verdict.Answer) addons.Verdict {
	return addons.Verdict{Known: a.Known, Score: a.Score, Adult: a.Adult, Source: string(a.Source), Detail: a.Detail, TimedOut: a.TimedOut, Unavailable: a.Unavailable}
}

func (c *pipelineClassifier) ClassifyText(ctx context.Context, req addons.TextRequest) addons.Verdict {
	return toVerdict(c.vs.Text(ctx, verdict.TextRequest{URL: req.URL, Title: req.Title, Text: req.Text, Budget: req.Budget}))
}

func (c *pipelineClassifier) ClassifyImage(ctx context.Context, req addons.ImageRequest) addons.Verdict {
	return toVerdict(c.vs.Image(ctx, verdict.ImageRequest{URL: req.URL, Data: req.Data, Budget: req.Budget}))
}

func (c *pipelineClassifier) ClassifyHost(ctx context.Context, req addons.HostRequest) addons.Verdict {
	return toVerdict(c.vs.Host(ctx, verdict.HostRequest{Host: req.Host, SamplePaths: req.SamplePaths, Budget: req.Budget}))
}

// ---- mgmtapi.ContentScanner (Tools page) ----

// ErrMLNotReady is returned by the scanner while the models are down.
var ErrMLNotReady = errors.New("classification models are not loaded")

type scanner struct{ st *MLStack }

func (s *scanner) ScanText(ctx context.Context, text string) (mgmtapi.TextScan, error) {
	if !s.st.Svc.Ready() {
		return mgmtapi.TextScan{}, ErrMLNotReady
	}
	a := s.st.Verdicts.Text(ctx, verdict.TextRequest{URL: "", Title: "", Text: text, Budget: 60 * time.Second})
	if !a.Known {
		return mgmtapi.TextScan{}, fmt.Errorf("no verdict: %s", a.Detail)
	}
	return mgmtapi.TextScan{Adult: a.Adult, Score: a.Score, Categories: labelsOf(a.Detail), Source: string(a.Source)}, nil
}

// labelsOf turns describe's "porn 0.82, sexy 0.10" back into labels.
func labelsOf(detail string) []string {
	var out []string
	for _, part := range strings.Split(detail, ", ") {
		if label, _, ok := strings.Cut(part, " "); ok {
			out = append(out, label)
		}
	}
	return out
}

func (s *scanner) ScanImage(ctx context.Context, img []byte) (mgmtapi.ImageScan, error) {
	if !s.st.Svc.Ready() {
		return mgmtapi.ImageScan{}, ErrMLNotReady
	}
	if _, err := imageprep.Prepare(img, s.st.maxPx); err != nil {
		return mgmtapi.ImageScan{}, err
	}
	// The Tools page wants every class, so score directly; the verdict
	// cache keeps only the summary.
	sc, err := s.st.Svc.Image(ctx, img)
	if err != nil {
		return mgmtapi.ImageScan{}, err
	}
	a := s.st.Verdicts.Image(ctx, verdict.ImageRequest{URL: "tools-scan", Data: img, Budget: 60 * time.Second})
	out := mgmtapi.ImageScan{Adult: a.Adult, Score: a.Score, Source: string(a.Source)}
	if !a.Known {
		out.Score, out.Source = sc.Unsafe, string(verdict.SourceModel)
	}
	for i, l := range sc.Labels {
		out.Detections = append(out.Detections, map[string]any{"class": l, "score": sc.Probs[i]})
	}
	return out, nil
}

func (s *scanner) Health(ctx context.Context) mgmtapi.ScannerHealth {
	st := s.st.Svc.Status()
	var ids []string
	for _, m := range st.Models {
		ids = append(ids, m.ID)
	}
	h := mgmtapi.ScannerHealth{Model: strings.Join(ids, ", "), Status: string(st.Phase), Detail: st.LastError}
	h.Available = st.Phase == ml.PhaseReady
	if h.Available && h.Detail == "" {
		h.Detail = "models loaded and answering"
	}
	return h
}

// ---- mgmtapi.MLController ----

type controller struct {
	svc *ml.Service
	vs  *verdict.Service
}

func (c *controller) Status() any {
	st := c.svc.Status()
	return struct {
		ml.Status
		QueueDepth int           `json:"queue_depth"`
		Cache      verdict.Stats `json:"cache"`
	}{st, c.vs.QueueDepth(), c.vs.Store().Stats()}
}
func (c *controller) Catalog() any    { return catalog.All() }
func (c *controller) CancelDownload() { c.svc.CancelDownload() }
func (c *controller) Download(ctx context.Context, model string) error {
	return c.svc.Download(ctx, strings.TrimSpace(model))
}
func (c *controller) Restart(ctx context.Context) error { return c.svc.Restart(ctx) }
func (c *controller) RemoveModel(id string) error       { return c.svc.RemoveModel(id) }
func (c *controller) LogTail(n int64) string            { return c.svc.LogTail(n) }

// ---- mgmtapi.DecisionStore ----

type decisions struct{ vs *verdict.Service }

func (d *decisions) List(kind, query string, limit int) (any, error) {
	return d.vs.Store().List(verdict.Kind(kind), query, limit)
}
func (d *decisions) Override(kind, key string, adult bool, note string) error {
	return d.vs.Override(verdict.Kind(kind), key, adult, note)
}
func (d *decisions) OverrideCategory(key, category, note string) error {
	return d.vs.OverrideCategory(key, category, note)
}
func (d *decisions) Delete(kind, key string) error {
	return d.vs.Store().Delete(verdict.Kind(kind), key)
}
func (d *decisions) Clear(kind string, includeManual bool) error {
	return d.vs.Store().Clear(verdict.Kind(kind), includeManual)
}
func (d *decisions) Stats() any { return d.vs.Store().Stats() }
