package mgmtapi

import (
	"context"
	"errors"
	"net/http"
	"strings"

	"github.com/go-chi/chi/v5"
)

// MLController is what the /api/ml/* routes drive. The process that owns
// the models (`run`) sets Server.ML; nil means the endpoints answer 503
// with a reason, which is the standalone-`mgmt` case.
type MLController interface {
	// Status returns the service's JSON-serialisable status payload.
	Status() any
	// Catalog lists the models the service can install.
	Catalog() any
	// Download starts fetching the runtime and the named model (empty =
	// every configured model) in the background.
	Download(ctx context.Context, model string) error
	// CancelDownload aborts a running download.
	CancelDownload()
	// Restart reloads the models.
	Restart(ctx context.Context) error
	// RemoveModel deletes an installed model that is not in use.
	RemoveModel(id string) error
	// LogTail returns the last n bytes of the service log.
	LogTail(n int64) string
}

func (s *Server) registerMLRoutes(r chi.Router) {
	r.Get("/api/ml/status", s.handleMLStatus)
	r.Get("/api/ml/catalog", s.handleMLCatalog)
	r.Get("/api/ml/log", s.handleMLLog)
	// Downloads write to disk and reloads interrupt service, so they are
	// gated with the other configuration mutations.
	r.With(s.requireUnlocked).Post("/api/ml/download", s.handleMLDownload)
	r.With(s.requireUnlocked).Post("/api/ml/download/cancel", s.handleMLDownloadCancel)
	r.With(s.requireUnlocked).Post("/api/ml/restart", s.handleMLRestart)
	r.With(s.requireUnlocked).Delete("/api/ml/models/{id}", s.handleMLRemoveModel)
}

func (s *Server) mlOr503(w http.ResponseWriter) (MLController, bool) {
	if s.ML == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "The models are not managed by this process (run `webfilter run`, not `webfilter mgmt`).")
		return nil, false
	}
	return s.ML, true
}

func (s *Server) handleMLStatus(w http.ResponseWriter, r *http.Request) {
	if s.ML == nil {
		writeJSON(w, http.StatusOK, map[string]any{
			"enabled": s.Settings().ML.Enabled, "phase": "unmanaged",
			"last_error": "The models are not managed by this process.",
		})
		return
	}
	writeJSON(w, http.StatusOK, s.ML.Status())
}

func (s *Server) handleMLCatalog(w http.ResponseWriter, r *http.Request) {
	c, ok := s.mlOr503(w)
	if !ok {
		return
	}
	writeJSON(w, http.StatusOK, c.Catalog())
}

func (s *Server) handleMLLog(w http.ResponseWriter, r *http.Request) {
	c, ok := s.mlOr503(w)
	if !ok {
		return
	}
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	_, _ = w.Write([]byte(c.LogTail(64 << 10)))
}

func (s *Server) handleMLDownload(w http.ResponseWriter, r *http.Request) {
	c, ok := s.mlOr503(w)
	if !ok {
		return
	}
	var payload struct {
		Model string `json:"model"`
	}
	_ = readJSON(r, &payload)
	// The download outlives this request: detach from r.Context().
	if err := c.Download(context.WithoutCancel(r.Context()), strings.TrimSpace(payload.Model)); err != nil {
		status := http.StatusBadRequest
		if errors.Is(err, errBusy) || strings.Contains(err.Error(), "already in progress") {
			status = http.StatusConflict
		}
		writeJSONError(w, status, err.Error())
		return
	}
	writeJSON(w, http.StatusAccepted, map[string]any{"status": "started"})
}

var errBusy = errors.New("busy")

func (s *Server) handleMLDownloadCancel(w http.ResponseWriter, r *http.Request) {
	c, ok := s.mlOr503(w)
	if !ok {
		return
	}
	c.CancelDownload()
	writeJSON(w, http.StatusOK, map[string]any{"status": "cancelled"})
}

func (s *Server) handleMLRestart(w http.ResponseWriter, r *http.Request) {
	c, ok := s.mlOr503(w)
	if !ok {
		return
	}
	if err := c.Restart(context.WithoutCancel(r.Context())); err != nil {
		writeJSONError(w, http.StatusBadGateway, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "restarted"})
}

func (s *Server) handleMLRemoveModel(w http.ResponseWriter, r *http.Request) {
	c, ok := s.mlOr503(w)
	if !ok {
		return
	}
	if err := c.RemoveModel(chi.URLParam(r, "id")); err != nil {
		writeJSONError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "removed"})
}
