package app

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/yjlion/onnx-web-filter/internal/classify/verdict"
	"github.com/yjlion/onnx-web-filter/internal/proxy/state"
)

// pageBackend answers every page "news" and every site "shopping".
type pageBackend struct{}

func (pageBackend) Ready() bool     { return true }
func (pageBackend) ModelID() string { return "fake" }
func (pageBackend) Vision() bool    { return false }
func (pageBackend) Image(context.Context, string, []byte, string) (verdict.Result, error) {
	return verdict.Result{}, errors.New("no images")
}
func (pageBackend) Text(context.Context, string, string, string) (verdict.Result, error) {
	return verdict.Result{}, errors.New("no text")
}
func (pageBackend) Host(context.Context, string, []string) (verdict.Result, error) {
	return verdict.Result{}, errors.New("no hosts")
}
func (pageBackend) Site(context.Context, string, string, string) (verdict.Result, error) {
	return verdict.Result{Category: "shopping", Confidence: 0.9}, nil
}
func (pageBackend) Page(context.Context, string, string, string, string) (verdict.Result, error) {
	return verdict.Result{Category: "news", Confidence: 0.9}, nil
}

func TestSiteCategorizerPageLookups(t *testing.T) {
	st, err := verdict.OpenMemory()
	if err != nil {
		t.Fatal(err)
	}
	vs := verdict.New(st, pageBackend{}, verdict.Options{Workers: 1})
	t.Cleanup(func() { vs.Close(); st.Close() })
	c := NewSiteCategorizer(vs, nil)
	ctx := context.Background()
	const u = "https://shop.example/article"

	// With content: the page is judged.
	a := c.Categorize(ctx, state.CategoryLookup{Host: "shop.example", URL: u, Text: "story", Enqueue: true, Budget: time.Second})
	if a.Scope != "page" || a.Category != "news" {
		t.Fatalf("page judged = %+v", a)
	}
	// Before a fetch the model's page verdict does not answer: the proxy
	// must fetch the page to see whether it changed.
	a = c.Categorize(ctx, state.CategoryLookup{Host: "shop.example", URL: u, Enqueue: true, Budget: time.Second})
	if a.Scope != "site" || a.Category != "shopping" {
		t.Fatalf("pre-fetch = %+v", a)
	}
	// The lookup tool asks for the cached verdict explicitly.
	a = c.Categorize(ctx, state.CategoryLookup{Host: "shop.example", URL: u, Cached: true})
	if a.Scope != "page" || a.Category != "news" {
		t.Fatalf("cached lookup = %+v", a)
	}
	// A manual page override answers before the fetch.
	if err := vs.OverridePageCategory(u, "education", ""); err != nil {
		t.Fatal(err)
	}
	a = c.Categorize(ctx, state.CategoryLookup{Host: "shop.example", URL: u, Enqueue: true})
	if a.Scope != "page" || a.Category != "education" || a.Source != "manual" {
		t.Fatalf("manual pre-fetch = %+v", a)
	}
}
