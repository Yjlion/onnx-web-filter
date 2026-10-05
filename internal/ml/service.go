// Package ml ties the ONNX Runtime install, the model catalog and the
// classifiers into one Service the rest of the filter talks to: "are the
// models loaded", "download them", "score this image".
//
// Everything runs in this process: the runtime is a shared library loaded
// once, and each model is an ONNX Runtime session.
package ml

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/yjlion/onnx-web-filter/internal/metrics"
	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/classify"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortenv"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
	"github.com/yjlion/onnx-web-filter/internal/models"
	"github.com/yjlion/onnx-web-filter/internal/sitecat"
)

// Phase is the service's coarse state for the status page.
type Phase string

const (
	PhaseDisabled      Phase = "disabled"
	PhaseNeedsDownload Phase = "needs_download"
	PhaseDownloading   Phase = "downloading"
	PhaseStarting      Phase = "starting"
	PhaseReady         Phase = "ready"
	PhaseFailed        Phase = "failed"
	PhaseStopped       Phase = "stopped"
)

// ErrNotReady is returned while the models are not loaded.
var ErrNotReady = errors.New("models are not loaded")

// ErrBusy is returned when a download is already running.
var ErrBusy = errors.New("a download is already in progress")

// Tasks lists the model tasks in display order.
var Tasks = []catalog.Task{catalog.TaskImage, catalog.TaskText, catalog.TaskSite}

// Service is the process-wide classification facade.
type Service struct {
	cfg    models.MLConfig
	hub    *catalog.Hub
	models map[catalog.Task]catalog.Model

	mu      sync.Mutex
	accel   ortrt.Accel
	phase   Phase
	lastErr string
	rt      *ortenv.Info
	dl      *download
	warmMs  int
	events  []string

	// handles holds the open models; inference takes the read lock, so
	// Stop can wait for in-flight calls before closing sessions.
	handles sync.RWMutex
	image   *classify.ImageClassifier
	text    *classify.TextClassifier
	emb     *classify.Embedder
	site    *classify.SiteClassifier
}

type download struct {
	what     string
	prog     *catalog.Progress
	step     string
	started  time.Time
	done     bool
	err      string
	cancel   context.CancelFunc
	finished time.Time
}

// New builds a Service from settings. Nothing loads until Start.
func New(cfg models.MLConfig) *Service {
	s := &Service{cfg: cfg, hub: catalog.NewHub(), accel: ortrt.ResolveAccel(cfg.Accel), models: map[catalog.Task]catalog.Model{}}
	s.phase = PhaseStopped
	if !cfg.Enabled {
		s.phase = PhaseDisabled
	}
	ids := map[catalog.Task]string{catalog.TaskImage: cfg.ImageModel, catalog.TaskText: cfg.TextModel, catalog.TaskSite: cfg.SiteModel}
	for _, t := range Tasks {
		m, err := catalog.Resolve(t, ids[t])
		if err != nil {
			m = catalog.Default(t)
			s.lastErr = fmt.Sprintf("%v in settings; using %s", err, m.ID)
		}
		s.models[t] = m
	}
	return s
}

// Config returns the settings the service was built from.
func (s *Service) Config() models.MLConfig { return s.cfg }

// Model returns the configured model for a task.
func (s *Service) Model(t catalog.Task) catalog.Model { return s.models[t] }

// ModelIDs names the configured models, for attributing cached verdicts.
func (s *Service) ModelIDs() string {
	return s.models[catalog.TaskImage].ID + "," + s.models[catalog.TaskText].ID + "," + s.models[catalog.TaskSite].ID
}

// Slots is the number of verdicts to run at once: the configured value, or
// 4 with a GPU and 2 on a CPU.
func (s *Service) Slots() int {
	if s.cfg.Parallel > 0 {
		return s.cfg.Parallel
	}
	if s.accel.GPU() {
		return 4
	}
	return 2
}

// Threads is the intra-op thread count per session.
func (s *Service) Threads() int {
	if s.cfg.Threads > 0 {
		return s.cfg.Threads
	}
	return max(1, runtime.NumCPU()/s.Slots())
}

// Accel is the resolved runtime flavour.
func (s *Service) Accel() ortrt.Accel {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.accel
}

// Missing reports which pieces still need downloading.
func (s *Service) Missing() (runtimeMissing bool, modelsMissing []string) {
	if _, err := ortenv.Locate(s.cfg.ORTLibPath, s.cfg.DataDir, s.Accel()); err != nil {
		runtimeMissing = true
	}
	for _, t := range Tasks {
		if _, ok := catalog.LoadInstalled(s.cfg.DataDir, s.models[t]); !ok {
			modelsMissing = append(modelsMissing, s.models[t].ID)
		}
	}
	return runtimeMissing, modelsMissing
}

// Start loads the runtime and models if they are on disk. When something
// is missing it does not download (that belongs behind an explicit action
// or the setup wizard): the phase becomes needs_download, and a later
// Download loads everything when it finishes.
func (s *Service) Start(ctx context.Context) error {
	if !s.cfg.Enabled {
		return nil
	}
	if rt, missing := s.Missing(); rt || len(missing) > 0 {
		s.setPhase(PhaseNeedsDownload, "")
		return nil
	}
	return s.load(ctx)
}

func (s *Service) load(ctx context.Context) error {
	s.setPhase(PhaseStarting, "")
	err := s.loadModels(ctx)
	if err != nil {
		s.closeModels()
		s.setPhase(PhaseFailed, err.Error())
		s.event("load failed: " + err.Error())
		return err
	}
	s.setPhase(PhaseReady, "")
	s.warmup(ctx)
	return nil
}

func (s *Service) loadModels(ctx context.Context) error {
	lib, err := ortenv.Locate(s.cfg.ORTLibPath, s.cfg.DataDir, s.Accel())
	if err != nil {
		return err
	}
	info, err := ortenv.Init(lib)
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.rt = &info
	s.mu.Unlock()
	s.event(fmt.Sprintf("ONNX Runtime %s loaded from %s (providers %s)", info.Version, info.Library.Path, strings.Join(info.Providers, ", ")))

	opts := classify.Options{Threads: s.Threads(), GPU: s.Accel().GPU() && info.HasProvider("CUDAExecutionProvider")}
	inst := func(t catalog.Task) (catalog.Model, catalog.Installed, error) {
		m := s.models[t]
		in, ok := catalog.LoadInstalled(s.cfg.DataDir, m)
		if !ok {
			return m, in, fmt.Errorf("%w: %s", catalog.ErrNotInstalled, m.ID)
		}
		return m, in, nil
	}

	s.handles.Lock()
	defer s.handles.Unlock()
	m, in, err := inst(catalog.TaskImage)
	if err != nil {
		return err
	}
	if s.image, err = classify.NewImageClassifier(m, in, opts); err != nil {
		return err
	}
	s.event(fmt.Sprintf("image model %s loaded (%s)", m.ID, s.image.Provider()))
	if m, in, err = inst(catalog.TaskText); err != nil {
		return err
	}
	if s.text, err = classify.NewTextClassifier(m, in, opts); err != nil {
		return err
	}
	s.event(fmt.Sprintf("text model %s loaded (%s)", m.ID, s.text.Provider()))
	if m, in, err = inst(catalog.TaskSite); err != nil {
		return err
	}
	if s.emb, err = classify.NewEmbedder(m, in, opts); err != nil {
		return err
	}
	var protos []classify.Prototype
	for _, p := range sitecat.PrototypeTexts() {
		protos = append(protos, classify.Prototype{Slug: p.Slug, Texts: p.Texts})
	}
	started := time.Now()
	if s.site, err = classify.NewSiteClassifier(ctx, s.emb, protos, 0); err != nil {
		return err
	}
	s.event(fmt.Sprintf("site model %s loaded (%s); %d category prototypes embedded in %d ms",
		m.ID, s.emb.Provider(), len(protos), time.Since(started).Milliseconds()))
	return nil
}

// warmup runs each model once so the first real verdict does not pay for
// lazy initialisation, and records how long that took.
func (s *Service) warmup(ctx context.Context) {
	started := time.Now()
	_, err := s.Text(ctx, "Example Domain. This domain is for use in illustrative examples in documents.")
	if err == nil {
		_, err = s.Site(ctx, "example.com", "Example Domain", "")
	}
	ms := int(time.Since(started).Milliseconds())
	s.mu.Lock()
	s.warmMs = ms
	s.mu.Unlock()
	if err != nil {
		s.event("warm-up failed: " + err.Error())
		return
	}
	s.event(fmt.Sprintf("warm-up complete in %d ms", ms))
}

func (s *Service) closeModels() {
	s.handles.Lock()
	defer s.handles.Unlock()
	if s.image != nil {
		_ = s.image.Close()
	}
	if s.text != nil {
		_ = s.text.Close()
	}
	if s.emb != nil {
		_ = s.emb.Close()
	}
	s.image, s.text, s.emb, s.site = nil, nil, nil, nil
}

// Ready reports whether verdicts can be requested right now.
func (s *Service) Ready() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.phase == PhaseReady
}

// Stop closes the models. The runtime library stays loaded: a process can
// load it only once.
func (s *Service) Stop() {
	s.closeModels()
	s.mu.Lock()
	if s.phase != PhaseDisabled {
		s.phase = PhaseStopped
	}
	s.mu.Unlock()
}

// Restart reloads the models.
func (s *Service) Restart(ctx context.Context) error {
	s.Stop()
	s.event("restart requested")
	return s.Start(ctx)
}

// Image scores encoded image bytes.
func (s *Service) Image(ctx context.Context, data []byte) (classify.Scores, error) {
	started := time.Now()
	s.handles.RLock()
	defer s.handles.RUnlock()
	if s.image == nil {
		observe("image", started, ErrNotReady)
		return classify.Scores{}, ErrNotReady
	}
	sc, err := s.image.Classify(ctx, data)
	observe("image", started, err)
	return sc, err
}

// Text scores page text.
func (s *Service) Text(ctx context.Context, text string) (classify.Scores, error) {
	started := time.Now()
	s.handles.RLock()
	defer s.handles.RUnlock()
	if s.text == nil {
		observe("text", started, ErrNotReady)
		return classify.Scores{}, ErrNotReady
	}
	sc, err := s.text.Classify(ctx, text)
	observe("text", started, err)
	return sc, err
}

// Site ranks the categories for a site.
func (s *Service) Site(ctx context.Context, host, title, description string) (classify.SiteScores, error) {
	started := time.Now()
	s.handles.RLock()
	defer s.handles.RUnlock()
	if s.site == nil {
		observe("category", started, ErrNotReady)
		return classify.SiteScores{}, ErrNotReady
	}
	sc, err := s.site.Classify(ctx, host, title, description)
	observe("category", started, err)
	return sc, err
}

func observe(kind string, started time.Time, err error) {
	result := "ok"
	switch {
	case errors.Is(err, ErrNotReady):
		result = "unavailable"
	case errors.Is(err, context.DeadlineExceeded):
		result = "timeout"
	case err != nil:
		result = "error"
	}
	metrics.MLRequests.Inc(kind, result)
	if err == nil {
		metrics.MLDuration.Observe(time.Since(started).Seconds(), kind)
	}
}

// Download fetches, in the background, the runtime and either one model or
// (modelID empty) every configured model, then loads everything if the
// service was waiting for it. ctx is the long-lived process context; the
// download has its own cancel.
func (s *Service) Download(ctx context.Context, modelID string) error {
	var want []catalog.Model
	what := "runtime and models"
	if modelID != "" {
		m, ok := catalog.Lookup(modelID)
		if !ok {
			return fmt.Errorf("unknown model %q", modelID)
		}
		want, what = []catalog.Model{m}, m.ID
	} else {
		for _, t := range Tasks {
			want = append(want, s.models[t])
		}
	}
	s.mu.Lock()
	if s.dl != nil && !s.dl.done {
		s.mu.Unlock()
		return ErrBusy
	}
	dctx, cancel := context.WithCancel(ctx)
	d := &download{what: what, prog: &catalog.Progress{}, started: time.Now(), cancel: cancel}
	s.dl = d
	if s.phase == PhaseNeedsDownload || s.phase == PhaseStopped || s.phase == PhaseFailed {
		s.phase = PhaseDownloading
	}
	s.mu.Unlock()
	s.event("download started: " + what)

	go func() {
		defer cancel()
		err := s.runDownload(dctx, d, want)
		s.mu.Lock()
		d.done, d.finished = true, time.Now()
		if err != nil {
			d.err = err.Error()
			s.lastErr = err.Error()
		}
		waiting := s.phase == PhaseDownloading && s.cfg.Enabled
		if waiting {
			s.phase = PhaseStopped
		}
		s.mu.Unlock()
		if err != nil {
			s.event("download failed: " + err.Error())
		} else {
			s.event("download finished: " + what)
		}
		if waiting {
			if err := s.Start(ctx); err != nil {
				slog.Error("ml: load after download failed", "err", err)
			}
		}
	}()
	return nil
}

func (s *Service) runDownload(ctx context.Context, d *download, want []catalog.Model) error {
	if err := os.MkdirAll(s.cfg.DataDir, 0o755); err != nil {
		return err
	}
	if _, err := ortenv.Locate(s.cfg.ORTLibPath, s.cfg.DataDir, s.Accel()); err != nil {
		accel := s.Accel()
		step := func(st string) { s.setStep(d, "runtime: "+st) }
		_, err := ortrt.Download(ctx, s.cfg.DataDir, ortrt.PinnedVersion, accel, step)
		if err != nil && accel.GPU() {
			// A GPU build that cannot be fetched should not strand the user.
			s.event(fmt.Sprintf("%s runtime download failed (%v); using the CPU build", accel, err))
			s.mu.Lock()
			s.accel = ortrt.AccelCPU
			s.mu.Unlock()
			_, err = ortrt.Download(ctx, s.cfg.DataDir, ortrt.PinnedVersion, ortrt.AccelCPU, step)
		}
		if err != nil {
			return fmt.Errorf("runtime: %w", err)
		}
	}
	for _, m := range want {
		if _, ok := catalog.LoadInstalled(s.cfg.DataDir, m); ok {
			continue
		}
		s.setStep(d, "model: "+m.ID)
		if _, err := s.hub.Download(ctx, s.cfg.DataDir, m, d.prog); err != nil {
			return fmt.Errorf("model %s: %w", m.ID, err)
		}
	}
	s.setStep(d, "done")
	return nil
}

func (s *Service) setStep(d *download, step string) {
	s.mu.Lock()
	d.step = step
	s.mu.Unlock()
}

// CancelDownload aborts a running download.
func (s *Service) CancelDownload() {
	s.mu.Lock()
	d := s.dl
	s.mu.Unlock()
	if d != nil && !d.done && d.cancel != nil {
		d.cancel()
	}
}

// RemoveModel deletes an installed model that is not configured.
func (s *Service) RemoveModel(id string) error {
	for _, t := range Tasks {
		if s.models[t].ID == id {
			return errors.New("cannot remove a model that is configured; pick another one in settings first")
		}
	}
	return catalog.Remove(s.cfg.DataDir, id)
}

func (s *Service) setPhase(p Phase, lastErr string) {
	s.mu.Lock()
	s.phase = p
	if lastErr != "" || p == PhaseReady {
		s.lastErr = lastErr
	}
	s.mu.Unlock()
}

// maxEvents bounds the service log shown on the Models page.
const maxEvents = 200

// event records a line in the service log and the process log.
func (s *Service) event(msg string) {
	slog.Info("ml: " + msg)
	line := time.Now().Format("2006-01-02 15:04:05") + "  " + msg
	s.mu.Lock()
	s.events = append(s.events, line)
	if len(s.events) > maxEvents {
		s.events = s.events[len(s.events)-maxEvents:]
	}
	s.mu.Unlock()
}

// LogTail returns the service log, at most n bytes from its end.
func (s *Service) LogTail(n int64) string {
	s.mu.Lock()
	out := strings.Join(s.events, "\n")
	s.mu.Unlock()
	if out != "" {
		out += "\n"
	}
	if int64(len(out)) > n {
		out = out[int64(len(out))-n:]
	}
	return out
}
