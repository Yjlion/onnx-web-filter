package ml

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/models"
)

func TestServiceWithoutDownloads(t *testing.T) {
	t.Setenv("ORT_LIB_PATH", "")
	cfg := models.NewMLConfig()
	cfg.DataDir = t.TempDir()
	cfg.Accel = "cpu"
	s := New(cfg)
	if err := s.Start(context.Background()); err != nil {
		t.Fatal(err)
	}
	st := s.Status()
	if st.Phase != PhaseNeedsDownload || s.Ready() {
		t.Fatalf("phase %s, ready %v", st.Phase, s.Ready())
	}
	if st.Runtime.Installed || len(st.Models) != 3 || st.Models[0].Installed {
		t.Errorf("status %+v", st)
	}
	if _, err := s.Image(context.Background(), []byte("x")); !errors.Is(err, ErrNotReady) {
		t.Errorf("Image before load: %v", err)
	}
	if _, err := s.Site(context.Background(), "example.com", "", ""); !errors.Is(err, ErrNotReady) {
		t.Errorf("Site before load: %v", err)
	}
	if err := s.RemoveModel(s.Model(catalog.TaskImage).ID); err == nil {
		t.Error("removing a configured model must fail")
	}
}

func TestServiceConfig(t *testing.T) {
	cfg := models.NewMLConfig()
	cfg.Accel = "cpu"
	cfg.TextModel = "nsfwjs-mobilenet" // an image model: falls back with an error
	s := New(cfg)
	if s.Model(catalog.TaskText).ID != catalog.Default(catalog.TaskText).ID {
		t.Errorf("text model %s", s.Model(catalog.TaskText).ID)
	}
	if !strings.Contains(s.Status().LastError, "not text") {
		t.Errorf("misconfigured model not reported: %q", s.Status().LastError)
	}
	if s.Slots() != 2 || s.Threads() < 1 {
		t.Errorf("slots %d threads %d", s.Slots(), s.Threads())
	}
	cfg.Enabled = false
	if st := New(cfg).Status(); st.Phase != PhaseDisabled {
		t.Errorf("disabled phase %s", st.Phase)
	}
}

// TestServiceLoadsRealModels needs ML_DATA_DIR (see classify's tests).
func TestServiceLoadsRealModels(t *testing.T) {
	dir := os.Getenv("ML_DATA_DIR")
	if dir == "" {
		t.Skip("ML_DATA_DIR not set")
	}
	cfg := models.NewMLConfig()
	cfg.DataDir = dir
	cfg.Accel = "cpu"
	s := New(cfg)
	ctx := context.Background()
	if err := s.Start(ctx); err != nil {
		t.Fatal(err)
	}
	defer s.Stop()
	if !s.Ready() {
		t.Fatalf("not ready: %+v", s.Status())
	}
	st := s.Status()
	if !st.Runtime.Loaded || st.WarmupMs <= 0 {
		t.Errorf("runtime %+v warmup %d", st.Runtime, st.WarmupMs)
	}
	for _, m := range st.Models {
		if !m.Loaded || m.Provider != "cpu" {
			t.Errorf("model %+v", m)
		}
	}
	adult, err := s.Text(ctx, "Hot naked girls stripping live on webcam, explicit hardcore porn videos free.")
	if err != nil || adult.Unsafe < 0.9 {
		t.Errorf("adult text: %+v %v", adult, err)
	}
	site, err := s.Site(ctx, "github.com", "GitHub", "")
	if err != nil || site.Category != "technology" {
		t.Errorf("github.com: %+v %v", site, err)
	}
	if !strings.Contains(s.LogTail(1<<20), "warm-up complete") {
		t.Errorf("log: %s", s.LogTail(1<<20))
	}

	// Restart closes and reopens the sessions; the runtime stays loaded.
	if err := s.Restart(ctx); err != nil || !s.Ready() {
		t.Fatalf("restart: %v ready=%v", err, s.Ready())
	}
	if _, err := s.Text(ctx, "hello"); err != nil {
		t.Errorf("after restart: %v", err)
	}
	s.Stop()
	if _, err := s.Text(ctx, "hello"); !errors.Is(err, ErrNotReady) {
		t.Errorf("after stop: %v", err)
	}
}
