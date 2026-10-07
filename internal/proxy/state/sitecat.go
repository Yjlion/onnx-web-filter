package state

import (
	"context"
	"time"
)

// CategoryLookup asks for a website's category (see internal/sitecat), or,
// with URL set, for one page's.
type CategoryLookup struct {
	Host               string
	Title, Description string
	// URL asks about one page. With content (Title, Description or Text)
	// the page is judged from it, again whenever it changes, and the answer
	// never falls back to the site's: when it is not Known, the site answer
	// stands. Without content only a manual override for the page answers
	// (or, with Cached, its cached verdict); otherwise the site's does. The
	// proxy has to fetch a page to see whether it changed, so a model's
	// page verdict is enforced on the response, never before the fetch.
	URL string
	// Cached lets a content-less page lookup answer from the page's cached
	// model verdict (the policy editor's lookup tool).
	Cached bool
	// Text is the page's visible text, for a page lookup.
	Text string
	// Enqueue lets the model be asked when nothing is known yet; without it
	// the lookup answers from the domain lists and the cache only.
	Enqueue bool
	// Budget is how long to wait for the model (0 = do not wait).
	Budget time.Duration
}

// CategoryAnswer is a categorizer's reply. When Known is false, TimedOut or
// Unavailable say why; neither is set for a cache-only miss.
type CategoryAnswer struct {
	// Scope is "page" for a page verdict, "site" otherwise.
	Scope       string
	Category    string
	Source      string
	Confidence  float64
	Known       bool
	TimedOut    bool
	Unavailable bool
}

// SiteCategorizer sorts websites into categories: the installed domain
// lists first, then the embedding model, with every verdict cached per site.
type SiteCategorizer interface {
	Categorize(ctx context.Context, q CategoryLookup) CategoryAnswer
}

type categorizerBox struct{ c SiteCategorizer }

// SetSiteCategorizer installs the categorizer (nil removes it). Safe to call
// while flows are running.
func (rt *Runtime) SetSiteCategorizer(c SiteCategorizer) {
	rt.categorizer.Store(&categorizerBox{c})
}

// SiteCategorizer returns the installed categorizer, or nil.
func (rt *Runtime) SiteCategorizer() SiteCategorizer {
	if rt == nil {
		return nil
	}
	if b := rt.categorizer.Load(); b != nil {
		return b.c
	}
	return nil
}
