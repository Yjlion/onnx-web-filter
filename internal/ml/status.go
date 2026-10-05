package ml

import (
	"time"

	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
	"github.com/yjlion/onnx-web-filter/internal/models"
)

// DownloadStatus is the live download view.
type DownloadStatus struct {
	Active    bool    `json:"active"`
	What      string  `json:"what,omitempty"`
	Step      string  `json:"step,omitempty"`
	BytesDone int64   `json:"bytes_done"`
	BytesTot  int64   `json:"bytes_total"`
	File      string  `json:"file,omitempty"`
	Percent   float64 `json:"percent"`
	Error     string  `json:"error,omitempty"`
	Seconds   float64 `json:"seconds"`
}

// RuntimeStatus describes the ONNX Runtime install.
type RuntimeStatus struct {
	Version   string   `json:"version"` // pinned, or the loaded library's
	Installed bool     `json:"installed"`
	Loaded    bool     `json:"loaded"`
	Library   string   `json:"library,omitempty"`
	Source    string   `json:"source,omitempty"`
	Providers []string `json:"providers,omitempty"`
}

// ModelStatus is one configured model.
type ModelStatus struct {
	Task      catalog.Task `json:"task"`
	ID        string       `json:"id"`
	Name      string       `json:"name"`
	License   string       `json:"license"`
	SizeMB    int64        `json:"size_mb"`
	Installed bool         `json:"installed"`
	Loaded    bool         `json:"loaded"`
	Provider  string       `json:"provider,omitempty"`
}

// Status is the full status payload for the UI and CLI.
type Status struct {
	Enabled   bool                `json:"enabled"`
	Phase     Phase               `json:"phase"`
	Accel     string              `json:"accel"`
	DataDir   string              `json:"data_dir"`
	Slots     int                 `json:"parallel"`
	Threads   int                 `json:"threads"`
	Runtime   RuntimeStatus       `json:"runtime"`
	Models    []ModelStatus       `json:"models"`
	Download  DownloadStatus      `json:"download"`
	LastError string              `json:"last_error,omitempty"`
	WarmupMs  int                 `json:"warmup_ms,omitempty"`
	Installed []catalog.Installed `json:"installed_models"`
	Budget    models.MLBudget     `json:"budget"`
}

// Status snapshots the service.
func (s *Service) Status() Status {
	rtMissing, _ := s.Missing()
	providers := map[catalog.Task]string{}
	s.handles.RLock()
	if s.image != nil {
		providers[catalog.TaskImage] = string(s.image.Provider())
	}
	if s.text != nil {
		providers[catalog.TaskText] = string(s.text.Provider())
	}
	if s.site != nil {
		providers[catalog.TaskSite] = string(s.emb.Provider())
	}
	s.handles.RUnlock()

	s.mu.Lock()
	defer s.mu.Unlock()
	st := Status{
		Enabled:   s.cfg.Enabled,
		Phase:     s.phase,
		Accel:     string(s.accel),
		DataDir:   s.cfg.DataDir,
		Slots:     s.Slots(),
		Threads:   s.Threads(),
		LastError: s.lastErr,
		WarmupMs:  s.warmMs,
		Installed: catalog.ListInstalled(s.cfg.DataDir),
		Budget:    s.cfg.Budget,
		Runtime:   RuntimeStatus{Version: ortrt.PinnedVersion, Installed: !rtMissing},
	}
	if st.Installed == nil {
		st.Installed = []catalog.Installed{}
	}
	if s.rt != nil {
		st.Runtime.Loaded = true
		st.Runtime.Version = s.rt.Version
		st.Runtime.Library = s.rt.Library.Path
		st.Runtime.Source = s.rt.Library.Source
		st.Runtime.Providers = s.rt.Providers
	}
	for _, t := range Tasks {
		m := s.models[t]
		_, inst := catalog.LoadInstalled(s.cfg.DataDir, m)
		st.Models = append(st.Models, ModelStatus{
			Task: t, ID: m.ID, Name: m.Name, License: m.License, SizeMB: catalog.MB(m.SizeBytes()),
			Installed: inst, Loaded: providers[t] != "", Provider: providers[t],
		})
	}
	if d := s.dl; d != nil {
		st.Download = DownloadStatus{
			Active: !d.done, What: d.what, Step: d.step, Error: d.err,
			BytesDone: d.prog.Done(), BytesTot: d.prog.Total(), File: d.prog.Current(),
		}
		end := d.finished
		if !d.done {
			end = time.Now()
		}
		st.Download.Seconds = end.Sub(d.started).Seconds()
		if st.Download.BytesTot > 0 {
			st.Download.Percent = 100 * float64(st.Download.BytesDone) / float64(st.Download.BytesTot)
		}
	}
	return st
}
