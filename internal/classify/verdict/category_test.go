package verdict

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"
	"time"
)

func TestCategoryListsThenCacheThenModel(t *testing.T) {
	b := &fakeBackend{}
	b.ready.Store(true)
	s := newTestService(t, b)
	ctx := context.Background()

	// Lists answer first, without the model.
	a := s.Category(ctx, CategoryRequest{Host: "www.shop.example", ListCategory: "gambling", Enqueue: true, Budget: time.Second})
	if !a.Known || a.Category != "gambling" || a.Source != SourceList || b.calls.Load() != 0 {
		t.Fatalf("list answer = %+v, calls %d", a, b.calls.Load())
	}

	// Cache-only lookup of an unknown site neither waits nor asks.
	a = s.Category(ctx, CategoryRequest{Host: "www.store.example"})
	if a.Known || a.TimedOut || a.Unavailable || b.calls.Load() != 0 {
		t.Fatalf("cache-only = %+v", a)
	}

	// The model answers within the budget and is cached per site.
	a = s.Category(ctx, CategoryRequest{Host: "www.store.example", Enqueue: true, Budget: time.Second})
	if !a.Known || a.Category != "shopping" || a.Source != SourceModel {
		t.Fatalf("model answer = %+v", a)
	}
	a = s.Category(ctx, CategoryRequest{Host: "m.store.example"})
	if !a.Known || !a.Cached || a.Category != "shopping" || b.calls.Load() != 1 {
		t.Fatalf("site cache = %+v calls %d", a, b.calls.Load())
	}

	// A low-confidence hostname verdict is refined once the title is known.
	_ = s.Category(ctx, CategoryRequest{Host: "www.store.example", Title: "Daily News", Enqueue: true})
	deadline := time.Now().Add(2 * time.Second)
	for {
		d, _ := s.Store().Get(KindCategory, "store.example")
		if d.Category == "news" && d.Detail == categoryFromPage {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("not refined: %+v", d)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestCategoryManualOverrideWins(t *testing.T) {
	b := &fakeBackend{}
	b.ready.Store(true)
	s := newTestService(t, b)
	if err := s.OverrideCategory("bank.example", "banking_finance", "mine"); err != nil {
		t.Fatal(err)
	}
	if err := s.OverrideCategory("x.example", "nope", ""); err == nil {
		t.Fatal("unknown slug accepted")
	}
	a := s.Category(context.Background(), CategoryRequest{Host: "www.bank.example", ListCategory: "phishing", Enqueue: true})
	if a.Category != "banking_finance" || a.Source != SourceManual {
		t.Fatalf("override = %+v", a)
	}
	// An exact-host override beats the site's.
	_ = s.OverrideCategory("games.bank.example", "gaming", "")
	a = s.Category(context.Background(), CategoryRequest{Host: "games.bank.example"})
	if a.Category != "gaming" {
		t.Fatalf("host override = %+v", a)
	}
}

func TestCategoryUnavailable(t *testing.T) {
	s := newTestService(t, &fakeBackend{})
	a := s.Category(context.Background(), CategoryRequest{Host: "a.example", Enqueue: true, Budget: time.Second})
	if !a.Unavailable {
		t.Fatalf("got %+v", a)
	}
}

func TestPageCategoryReclassifiesChangedContent(t *testing.T) {
	b := &fakeBackend{}
	b.ready.Store(true)
	s := newTestService(t, b)
	ctx := context.Background()
	u := "https://www.store.example/today?utm_source=x"

	// Nothing known before the page is seen.
	if a := s.CachedPageCategory(u); a.Known {
		t.Fatalf("cached before seen = %+v", a)
	}
	a := s.PageCategory(ctx, PageCategoryRequest{URL: u, Title: "Deals", Text: "buy now", Budget: time.Second})
	if !a.Known || a.Category != "shopping" || a.Kind != KindPage || a.Key != "www.store.example/today" || b.calls.Load() != 1 {
		t.Fatalf("first = %+v calls %d", a, b.calls.Load())
	}
	// Same content: cache, no model call.
	a = s.PageCategory(ctx, PageCategoryRequest{URL: u, Title: "Deals", Text: "buy  NOW", Budget: time.Second})
	if !a.Cached || a.Category != "shopping" || b.calls.Load() != 1 {
		t.Fatalf("same content = %+v calls %d", a, b.calls.Load())
	}
	// Changed content: asked again, the row is replaced.
	a = s.PageCategory(ctx, PageCategoryRequest{URL: u, Title: "Deals", Text: "election results", Budget: time.Second})
	if a.Cached || a.Category != "news" || b.calls.Load() != 2 {
		t.Fatalf("changed content = %+v calls %d", a, b.calls.Load())
	}
	if a = s.CachedPageCategory("http://www.store.example/today"); !a.Known || a.Category != "news" {
		t.Fatalf("cached after change = %+v", a)
	}
	// A page verdict does not touch the site's.
	if _, ok := s.Store().Get(KindCategory, "store.example"); ok {
		t.Fatal("page verdict leaked into the site row")
	}
	// No content: unknown, no call.
	if a = s.PageCategory(ctx, PageCategoryRequest{URL: "https://www.store.example/empty", Budget: time.Second}); a.Known || a.TimedOut || a.Unavailable {
		t.Fatalf("empty page = %+v", a)
	}
}

func TestPageCategoryManualOverridesWin(t *testing.T) {
	b := &fakeBackend{}
	b.ready.Store(true)
	s := newTestService(t, b)
	ctx := context.Background()
	if err := s.OverrideCategory("bank.example", "banking_finance", ""); err != nil {
		t.Fatal(err)
	}
	a := s.PageCategory(ctx, PageCategoryRequest{URL: "https://www.bank.example/x", Text: "election", Budget: time.Second})
	if a.Category != "banking_finance" || a.Source != SourceManual || b.calls.Load() != 0 {
		t.Fatalf("site override = %+v", a)
	}
	if err := s.OverridePageCategory("www.bank.example/blog", "news", ""); err != nil {
		t.Fatal(err)
	}
	a = s.PageCategory(ctx, PageCategoryRequest{URL: "https://www.bank.example/blog#top", Text: "rates", Budget: time.Second})
	if a.Category != "news" || a.Source != SourceManual {
		t.Fatalf("page override = %+v", a)
	}
	if a = s.CachedPageCategory("https://www.bank.example/blog"); a.Category != "news" {
		t.Fatalf("cached page override = %+v", a)
	}
}

func TestStorePrune(t *testing.T) {
	st, err := OpenMemory()
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	old := time.Now().Add(-48 * time.Hour)
	for _, d := range []Decision{
		{Kind: KindPage, Key: "a.example/old", Category: "news", Source: SourceModel, Created: old},
		{Kind: KindPage, Key: "a.example/manual", Category: "news", Source: SourceManual, Created: old},
		{Kind: KindPage, Key: "a.example/new", Category: "news", Source: SourceModel},
		{Kind: KindCategory, Key: "a.example", Category: "news", Source: SourceModel, Created: old},
	} {
		if err := st.Put(d); err != nil {
			t.Fatal(err)
		}
	}
	n, err := st.Prune(KindPage, time.Now().Add(-24*time.Hour))
	if err != nil || n != 1 {
		t.Fatalf("pruned %d, %v", n, err)
	}
	for key, want := range map[string]bool{"a.example/old": false, "a.example/manual": true, "a.example/new": true} {
		if _, ok := st.Get(KindPage, key); ok != want {
			t.Errorf("%s present = %v, want %v", key, ok, want)
		}
	}
	if _, ok := st.Get(KindCategory, "a.example"); !ok {
		t.Error("other kind pruned")
	}
}

func TestStoreMigratesOldSchema(t *testing.T) {
	path := filepath.Join(t.TempDir(), "old.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`CREATE TABLE decisions (kind TEXT NOT NULL, key TEXT NOT NULL, score REAL NOT NULL, adult INTEGER NOT NULL,
		source TEXT NOT NULL, model TEXT, detail TEXT, hint TEXT, confidence REAL NOT NULL DEFAULT 0, created INTEGER NOT NULL,
		hits INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (kind, key));
		INSERT INTO decisions VALUES('host','ads.example',0.9,1,'llm','m','ads','',0.9,1,0);`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	st, err := Open(path, 10)
	if err != nil {
		t.Fatal(err)
	}
	defer st.Close()
	if d, ok := st.Get(KindHost, "ads.example"); !ok || !d.Adult {
		t.Fatalf("old row lost: %+v %v", d, ok)
	}
	if err := st.Put(Decision{Kind: KindCategory, Key: "news.example", Category: "news", Source: SourceModel}); err != nil {
		t.Fatal(err)
	}
	st.lru.purge()
	if d, _ := st.Get(KindCategory, "news.example"); d.Category != "news" {
		t.Fatalf("category not persisted: %+v", d)
	}
	if err := st.Put(Decision{Kind: KindPage, Key: "news.example/a", Category: "news", Source: SourceModel, ContentHash: "h1"}); err != nil {
		t.Fatal(err)
	}
	st.lru.purge()
	if d, _ := st.Get(KindPage, "news.example/a"); d.ContentHash != "h1" {
		t.Fatalf("content hash not persisted: %+v", d)
	}
}
