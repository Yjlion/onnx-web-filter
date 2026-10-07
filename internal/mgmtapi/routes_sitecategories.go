package mgmtapi

import (
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/yjlion/onnx-web-filter/internal/proxy/state"
	"github.com/yjlion/onnx-web-filter/internal/sitecat"
)

// registerSiteCategoryRoutes wires the website taxonomy and a lookup that
// shows what category a site gets (and asks the model if it is new). Given
// a URL with a path, the lookup reports that page's verdict when one is
// cached (pages are only judged from their content, as the proxy sees it).
func (s *Server) registerSiteCategoryRoutes(r chi.Router) {
	r.Get("/api/sitecategories", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{"categories": sitecat.All(), "available": s.Sites != nil})
	})
	r.Get("/api/sitecategories/lookup", s.handleSiteCategoryLookup)
}

func (s *Server) handleSiteCategoryLookup(w http.ResponseWriter, r *http.Request) {
	raw := strings.TrimSpace(r.URL.Query().Get("host"))
	host := sitecat.HostOf(raw)
	if host == "" || strings.ContainsAny(host, " \t") {
		writeJSONError(w, http.StatusBadRequest, "host is required")
		return
	}
	if s.Sites == nil {
		writeJSONError(w, http.StatusServiceUnavailable, "Site categories are not available in this process (run `webfilter run`).")
		return
	}
	q := state.CategoryLookup{Host: host, Enqueue: true, Budget: 10 * time.Second}
	if strings.Contains(raw, "/") {
		q.URL, q.Cached = raw, true
	}
	ans := s.Sites.Categorize(r.Context(), q)
	out := map[string]any{"host": host, "site": sitecat.SiteKey(host), "known": ans.Known, "scope": ans.Scope}
	if ans.Scope == "page" {
		out["page"] = sitecat.PageKey(raw)
	}
	switch {
	case ans.Known:
		out["category"] = ans.Category
		out["label"] = sitecat.Label(ans.Category)
		out["source"] = ans.Source
		out["confidence"] = ans.Confidence
	case ans.TimedOut:
		out["detail"] = "the model is still deciding; try again in a moment"
	default:
		out["detail"] = "the model is not running"
	}
	writeJSON(w, http.StatusOK, out)
}
