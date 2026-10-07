package addons

import (
	"strings"

	"github.com/yjlion/onnx-web-filter/internal/adblock"
	"github.com/yjlion/onnx-web-filter/internal/models"
	"github.com/yjlion/onnx-web-filter/internal/proxy"
	"github.com/yjlion/onnx-web-filter/internal/proxy/state"
	"github.com/yjlion/onnx-web-filter/internal/sitecat"
)

// CategoryFilter blocks or allows websites and pages by category (shopping,
// news, social media, banking, ...). A site's category comes from the
// installed domain lists when they know the site, otherwise from the
// embedding model, cached per site (see the SiteCategorizer on the runtime).
// Every HTML page a navigation loads is also categorized from its own
// content, cached per page and judged again when its content changes; a
// page's verdict wins over its site's.
//
// Only top-level navigations ask the model and wait for it, within the
// policy's budget; a first visit that outlasts the budget gets on_timeout.
// The request phase goes by the site's category (or a manual override for
// the page); a site refused there is never fetched. The response phase
// judges the page that actually arrived, so a page refused for its own
// content is fetched and checked again on every load (a cache hit while it
// is unchanged) and is let through once it changes to an allowed category. Sub-resources
// are judged from what is already known about their site and never queue a
// model call, so one page's dozens of CDN and API hosts cannot flood it.
type CategoryFilter struct{}

func (CategoryFilter) Name() string { return "category_filter" }

func (CategoryFilter) HandleRequest(fc *proxy.FlowContext) {
	if fc.URLAllowed || fc.MitmPassthrough || fc.Policy == nil || fc.Response != nil {
		return
	}
	cfg := fc.Policy.CategoryFilter
	if !cfg.Enabled || (cfg.Mode != models.UrlFilterModeWhitelist && len(cfg.Categories) == 0) {
		return
	}
	nav := isNavigation(fc)
	host := strings.ToLower(fc.Request.URL.Hostname())

	cat := fc.Runtime.SiteCategorizer()
	var ans state.CategoryAnswer
	if cat == nil {
		ans.Unavailable = true
	} else {
		q := state.CategoryLookup{Host: host, Enqueue: nav}
		if nav {
			q.URL = fc.Request.URL.String()
			q.Budget = budgetFor(fc, cfg.BudgetMs, "category")
		}
		ans = cat.Categorize(fc.Request.Context(), q)
	}

	switch {
	case ans.Known:
		// In allow-only mode a sub-resource is never refused for its
		// category: the page it belongs to was already allowed, and the
		// hosts it pulls from are mostly CDNs the model may file as "other".
		if !nav && cfg.Mode == models.UrlFilterModeWhitelist {
			return
		}
		blockCategory(fc, cfg, ans)
	case !nav:
		// Unknown sub-resource: allow.
	case ans.TimedOut:
		if cfg.OnTimeout == models.FallbackBlock {
			fc.Block("Site not categorized yet (the model is still deciding; try again shortly)", "category_filter")
		}
	default:
		if cfg.OnUnavailable == models.FallbackBlock {
			fc.Block("Site category unavailable (the model is not running)", "category_filter")
		}
	}
}

// HandleResponse categorizes the page a navigation loaded from its own
// content and blocks it if that category is refused. A page whose verdict
// is cached for the same content costs a hash; new or changed content asks
// the model within the category budget. If the model cannot answer in
// time the request phase's decision stands and the verdict is cached for
// the next load.
func (CategoryFilter) HandleResponse(fc *proxy.FlowContext) {
	if fc.URLAllowed || fc.MitmPassthrough || fc.Policy == nil || fc.Response == nil || fc.WFAction == "blocked" {
		return
	}
	cfg := fc.Policy.CategoryFilter
	if !cfg.Enabled || (cfg.Mode != models.UrlFilterModeWhitelist && len(cfg.Categories) == 0) {
		return
	}
	if fc.Response.StatusCode < 200 || fc.Response.StatusCode > 299 ||
		!strings.Contains(fc.Response.Header.Get("Content-Type"), "text/html") || !isNavigation(fc) {
		return
	}
	cat := fc.Runtime.SiteCategorizer()
	if cat == nil {
		return
	}
	page := fc.Page()
	host := strings.ToLower(fc.Request.URL.Hostname())
	ctx := fc.Request.Context()
	// A site the model categorized from its hostname alone, unsure, gets a
	// second look now that its title is known (background, never waited
	// on). Sub-resources and tunnels still go by the site's category.
	if page.Title != "" || page.Description != "" {
		cat.Categorize(ctx, state.CategoryLookup{Host: host, Title: page.Title, Description: page.Description, Enqueue: true})
	}
	ans := cat.Categorize(ctx, state.CategoryLookup{
		Host: host, URL: fc.Request.URL.String(), Title: page.Title, Description: page.Description,
		Text:    strings.TrimSpace(strings.Join(page.Headings, " | ") + "\n" + page.Text),
		Enqueue: true, Budget: budgetFor(fc, cfg.BudgetMs, "category"),
	})
	if ans.Known {
		blockCategory(fc, cfg, ans)
	}
}

// blockCategory blocks fc if the policy refuses ans's category.
func blockCategory(fc *proxy.FlowContext, cfg models.CategoryFilterConfig, ans state.CategoryAnswer) {
	if !cfg.Blocks(ans.Category) {
		return
	}
	what := "Site"
	if ans.Scope == "page" {
		what = "Page"
	}
	if cfg.Mode == models.UrlFilterModeWhitelist {
		fc.Block(what+" category '"+sitecat.Label(ans.Category)+"' is not in the allowed categories", "category_filter")
	} else {
		fc.Block(what+" category '"+sitecat.Label(ans.Category)+"' blocked by policy", "category_filter")
	}
}

// isNavigation reports whether fc is a top-level page load.
func isNavigation(fc *proxy.FlowContext) bool {
	h := fc.Request.Header
	return adblock.TypeOf(fc.Request.URL, h.Get("Sec-Fetch-Dest"), h.Get("Accept"), h.Get("X-Requested-With")) == adblock.TypeDocument
}
