package addons_test

import (
	"context"
	"net/http"
	"strings"
	"sync"
	"testing"

	"github.com/yjlion/onnx-web-filter/internal/models"
	"github.com/yjlion/onnx-web-filter/internal/proxy"
	"github.com/yjlion/onnx-web-filter/internal/proxy/addons"
	"github.com/yjlion/onnx-web-filter/internal/proxy/state"
)

// fakeCategorizer answers from a fixed map and records every lookup.
type fakeCategorizer struct {
	mu      sync.Mutex
	known   map[string]string
	timeout bool
	down    bool
	// pages answers page lookups that carry content; cached answers those
	// without (the request phase: manual page overrides). Both by URL.
	pages  map[string]string
	cached map[string]string
	asked  []state.CategoryLookup
}

func (f *fakeCategorizer) Categorize(_ context.Context, q state.CategoryLookup) state.CategoryAnswer {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.asked = append(f.asked, q)
	if q.URL != "" {
		if q.Title+q.Description+q.Text != "" {
			if c, ok := f.pages[q.URL]; ok {
				return state.CategoryAnswer{Scope: "page", Category: c, Known: true, Source: "model"}
			}
			return state.CategoryAnswer{Scope: "page", TimedOut: true}
		}
		if c, ok := f.cached[q.URL]; ok {
			return state.CategoryAnswer{Scope: "page", Category: c, Known: true, Source: "model"}
		}
	}
	if c, ok := f.known[q.Host]; ok {
		return state.CategoryAnswer{Category: c, Known: true, Source: "model"}
	}
	if !q.Enqueue {
		return state.CategoryAnswer{}
	}
	if f.down {
		return state.CategoryAnswer{Unavailable: true}
	}
	return state.CategoryAnswer{TimedOut: f.timeout}
}

func categoryFlow(t *testing.T, cat *fakeCategorizer, url string, nav bool, cfg models.CategoryFilterConfig) *proxy.FlowContext {
	t.Helper()
	rt := newTestRuntime(t)
	rt.SetSiteCategorizer(cat)
	fc := newFlow(t, rt, url)
	if nav {
		fc.Request.Header.Set("Sec-Fetch-Dest", "document")
	} else {
		fc.Request.Header.Set("Sec-Fetch-Dest", "script")
	}
	p := models.NewPolicy()
	cfg.Enabled = true
	p.CategoryFilter = cfg
	fc.Policy = &p
	return fc
}

func blockList(cats ...string) models.CategoryFilterConfig {
	c := models.NewCategoryFilterConfig()
	c.Categories = cats
	return c
}

func TestCategoryFilterBlocksListedCategoryOnNavigation(t *testing.T) {
	cat := &fakeCategorizer{known: map[string]string{"shop.example": "shopping", "news.example": "news"}}
	fc := categoryFlow(t, cat, "http://shop.example/", true, blockList("shopping"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.WFAction != "blocked" || fc.WFComponent != "category_filter" {
		t.Fatalf("shopping not blocked: %q/%q", fc.WFAction, fc.WFComponent)
	}
	if len(cat.asked) != 1 || !cat.asked[0].Enqueue || cat.asked[0].Budget <= 0 {
		t.Fatalf("navigation lookup = %+v", cat.asked)
	}

	fc = categoryFlow(t, cat, "http://news.example/", true, blockList("shopping"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response != nil {
		t.Fatal("news blocked")
	}
}

func TestCategoryFilterSubresourcesNeverAskTheModel(t *testing.T) {
	cat := &fakeCategorizer{known: map[string]string{"social.example": "social_media"}}
	fc := categoryFlow(t, cat, "http://cdn.unknown.example/a.js", false, blockList("social_media"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response != nil || cat.asked[0].Enqueue || cat.asked[0].Budget != 0 {
		t.Fatalf("sub-resource lookup = %+v blocked=%v", cat.asked, fc.Response != nil)
	}
	// A known blocked category is still refused as a sub-resource.
	fc = categoryFlow(t, cat, "http://social.example/widget.js", false, blockList("social_media"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response == nil {
		t.Fatal("known social sub-resource allowed")
	}
}

func TestCategoryFilterWhitelist(t *testing.T) {
	cat := &fakeCategorizer{known: map[string]string{"edu.example": "education", "game.example": "gaming",
		"cdn.example": "infrastructure", "other.example": "other"}}
	cfg := blockList("education")
	cfg.Mode = models.UrlFilterModeWhitelist
	for url, wantBlocked := range map[string]bool{
		"http://edu.example/": false, "http://game.example/": true, "http://cdn.example/": false,
	} {
		fc := categoryFlow(t, cat, url, true, cfg)
		addons.CategoryFilter{}.HandleRequest(fc)
		if (fc.Response != nil) != wantBlocked {
			t.Errorf("%s blocked=%v want %v", url, fc.Response != nil, wantBlocked)
		}
	}
	// Sub-resources in allow-only mode pass whatever their category.
	fc := categoryFlow(t, cat, "http://other.example/x.js", false, cfg)
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response != nil {
		t.Error("whitelist blocked a sub-resource")
	}
}

func TestCategoryFilterFallbacks(t *testing.T) {
	cfg := blockList("shopping")
	cfg.OnTimeout = models.FallbackBlock
	fc := categoryFlow(t, &fakeCategorizer{timeout: true}, "http://new.example/", true, cfg)
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response == nil {
		t.Error("on_timeout=block did not block")
	}
	fc = categoryFlow(t, &fakeCategorizer{timeout: true}, "http://new.example/", true, blockList("shopping"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response != nil {
		t.Error("default on_timeout should allow")
	}
	cfg = blockList("shopping")
	cfg.OnUnavailable = models.FallbackBlock
	fc = categoryFlow(t, &fakeCategorizer{down: true}, "http://new.example/", true, cfg)
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response == nil {
		t.Error("on_unavailable=block did not block")
	}
}

func TestCategoryFilterAllowListWins(t *testing.T) {
	cat := &fakeCategorizer{known: map[string]string{"shop.example": "shopping"}}
	fc := categoryFlow(t, cat, "http://shop.example/", true, blockList("shopping"))
	fc.URLAllowed = true
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response != nil || len(cat.asked) != 0 {
		t.Fatal("allow-listed URL was categorized or blocked")
	}
}

// respond gives fc an HTML response, as the upstream fetch would.
func respond(fc *proxy.FlowContext, status int, contentType, body string) {
	fc.Response = &http.Response{StatusCode: status, Header: http.Header{"Content-Type": []string{contentType}}}
	fc.ResponseBody = []byte(body)
}

const newsPage = `<html><head><title>Election results</title></head><body><h1>Votes counted</h1><p>The results are in.</p></body></html>`

func TestCategoryFilterPageVerdictWinsOverSite(t *testing.T) {
	// The site is shopping (allowed); one of its pages is news (blocked).
	cat := &fakeCategorizer{known: map[string]string{"shop.example": "shopping"},
		pages: map[string]string{"http://shop.example/blog/vote": "news", "http://shop.example/": "shopping"}}
	fc := categoryFlow(t, cat, "http://shop.example/blog/vote", true, blockList("news"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response != nil {
		t.Fatal("request phase blocked an allowed site with no page verdict yet")
	}
	respond(fc, 200, "text/html; charset=utf-8", newsPage)
	addons.CategoryFilter{}.HandleResponse(fc)
	if fc.WFAction != "blocked" || !strings.Contains(string(fc.ResponseBody), "Page category") {
		t.Fatalf("news page not blocked: %q", fc.WFAction)
	}
	last := cat.asked[len(cat.asked)-1]
	if last.URL != "http://shop.example/blog/vote" || last.Title != "Election results" || !strings.Contains(last.Text, "Votes counted") ||
		!strings.Contains(last.Text, "The results are in.") || last.Budget <= 0 {
		t.Fatalf("page lookup = %+v", last)
	}

	// The shop's own home page passes.
	fc = categoryFlow(t, cat, "http://shop.example/", true, blockList("news"))
	addons.CategoryFilter{}.HandleRequest(fc)
	respond(fc, 200, "text/html", `<title>Deals</title><p>buy</p>`)
	addons.CategoryFilter{}.HandleResponse(fc)
	if fc.WFAction == "blocked" {
		t.Fatal("shopping page blocked")
	}
}

func TestCategoryFilterRequestUsesPageOverride(t *testing.T) {
	cat := &fakeCategorizer{known: map[string]string{"shop.example": "shopping"},
		cached: map[string]string{"http://shop.example/blog/vote": "news"}}
	fc := categoryFlow(t, cat, "http://shop.example/blog/vote", true, blockList("news"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.WFAction != "blocked" {
		t.Fatal("page overridden to news not blocked before fetch")
	}
	// A sub-resource never asks about its page.
	fc = categoryFlow(t, cat, "http://shop.example/blog/vote", false, blockList("news"))
	addons.CategoryFilter{}.HandleRequest(fc)
	if fc.Response != nil || cat.asked[len(cat.asked)-1].URL != "" {
		t.Fatalf("sub-resource lookup = %+v", cat.asked[len(cat.asked)-1])
	}
}

func TestCategoryFilterResponseSkips(t *testing.T) {
	cfg := blockList("news")
	cfg.OnTimeout = models.FallbackBlock
	for name, setup := range map[string]func(fc *proxy.FlowContext){
		// The model did not answer in time: the request phase's call stands.
		"timeout":      func(fc *proxy.FlowContext) { respond(fc, 200, "text/html", newsPage) },
		"not html":     func(fc *proxy.FlowContext) { respond(fc, 200, "application/json", `{}`) },
		"error status": func(fc *proxy.FlowContext) { respond(fc, 404, "text/html", newsPage) },
		"sub-resource": func(fc *proxy.FlowContext) {
			fc.Request.Header.Set("Sec-Fetch-Dest", "iframe")
			respond(fc, 200, "text/html", newsPage)
		},
	} {
		cat := &fakeCategorizer{pages: map[string]string{}}
		if name != "timeout" {
			cat.pages["http://new.example/a"] = "news"
		}
		fc := categoryFlow(t, cat, "http://new.example/a", true, cfg)
		setup(fc)
		addons.CategoryFilter{}.HandleResponse(fc)
		if fc.WFAction == "blocked" {
			t.Errorf("%s: blocked", name)
		}
	}
}
