package classify

import (
	"bufio"
	"context"
	"os"
	"strings"
	"testing"

	"github.com/yjlion/onnx-web-filter/internal/sitecat"
)

type siteCase struct{ host, title, slug string }

func loadSites(t testing.TB) []siteCase {
	f, err := os.Open("testdata/sites.tsv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []siteCase
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		p := strings.Split(line, "\t")
		if len(p) != 3 || !sitecat.Valid(p[2]) {
			t.Fatalf("bad line %q", line)
		}
		out = append(out, siteCase{p[0], p[1], p[2]})
	}
	return out
}

func siteClassifier(t testing.TB) *SiteClassifier {
	m, inst := installed(t, "minilm-l6")
	e, err := NewEmbedder(m, inst, Options{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	var protos []Prototype
	for _, p := range sitecat.PrototypeTexts() {
		protos = append(protos, Prototype{Slug: p.Slug, Texts: p.Texts})
	}
	c, err := NewSiteClassifier(context.Background(), e, protos, 0)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// TestSiteCategoryAccuracy reports accuracy on the held-out set and fails
// if it drops below the floor measured when the prototypes were written.
func TestSiteCategoryAccuracy(t *testing.T) {
	c := siteClassifier(t)
	cases := loadSites(t)
	ctx := context.Background()
	var withTitle, top3, hostOnly int
	var confRight, confWrong float64
	var nRight, nWrong, hostSure, hostSureRight int
	for _, s := range cases {
		r, err := c.Classify(ctx, s.host, s.title, "")
		if err != nil {
			t.Fatal(err)
		}
		if r.Category == s.slug {
			withTitle++
			confRight += r.Confidence
			nRight++
		} else {
			confWrong += r.Confidence
			nWrong++
			t.Logf("miss: %-32s %-45.45q want %-16s got %-16s (%.2f)", s.host, s.title, s.slug, r.Category, r.Confidence)
		}
		for _, rs := range r.Ranked[:3] {
			if rs.Slug == s.slug {
				top3++
			}
		}
		h, err := c.Classify(ctx, s.host, "", "")
		if err != nil {
			t.Fatal(err)
		}
		if h.Category == s.slug {
			hostOnly++
		}
		// Answers at or above the verdict service's re-ask threshold are
		// final, so their precision is what users see.
		if h.Confidence >= 0.6 {
			hostSure++
			if h.Category == s.slug {
				hostSureRight++
			}
		}
	}
	n := float64(len(cases))
	t.Logf("%d sites: top-1 %.0f%% (host only %.0f%%), top-3 %.0f%%; mean confidence right %.2f, wrong %.2f",
		len(cases), 100*float64(withTitle)/n, 100*float64(hostOnly)/n, 100*float64(top3)/n,
		confRight/max(1, float64(nRight)), confWrong/max(1, float64(nWrong)))
	t.Logf("host only, confidence >= 0.6: %d sites, %.0f%% right", hostSure, 100*float64(hostSureRight)/max(1, float64(hostSure)))
	if float64(withTitle)/n < siteAccuracyFloor {
		t.Errorf("top-1 accuracy %.2f below floor %.2f", float64(withTitle)/n, siteAccuracyFloor)
	}
}

const siteAccuracyFloor = 0.75

type pageCase struct{ url, title, text, slug string }

func loadPages(t testing.TB) []pageCase {
	f, err := os.Open("testdata/pages.tsv")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	var out []pageCase
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		p := strings.Split(line, "\t")
		if len(p) != 4 || !sitecat.Valid(p[3]) {
			t.Fatalf("bad line %q", line)
		}
		out = append(out, pageCase{p[0], p[1], p[2], p[3]})
	}
	return out
}

// TestPageCategoryAccuracy checks that a page judged from its own content
// lands in its own category, not its site's: every case is a page whose
// category differs from what its domain is known for. Measured when
// written: 64% from the page, 7% from the site; the floor leaves room for
// int8 outputs varying by CPU.
func TestPageCategoryAccuracy(t *testing.T) {
	c := siteClassifier(t)
	ctx := context.Background()
	cases := loadPages(t)
	var page, site int
	for _, p := range cases {
		r, err := c.ClassifyPage(ctx, p.url, p.title, "", p.text)
		if err != nil {
			t.Fatal(err)
		}
		if r.Category == p.slug {
			page++
		} else {
			t.Logf("miss: %-50.50s want %-12s got %-12s (%.2f)", p.url, p.slug, r.Category, r.Confidence)
		}
		h, err := c.Classify(ctx, sitecat.HostOf(p.url), "", "")
		if err != nil {
			t.Fatal(err)
		}
		if h.Category == p.slug {
			site++
		}
	}
	n := float64(len(cases))
	t.Logf("%d pages: page verdict %.0f%%, site verdict %.0f%%", len(cases), 100*float64(page)/n, 100*float64(site)/n)
	if float64(page)/n < 0.5 {
		t.Errorf("page accuracy %.0f%% below the 50%% floor", 100*float64(page)/n)
	}
}

func TestPageQueryText(t *testing.T) {
	got := PageQueryText("https://www.example.com/news/2026/election-results?id=7", " Results ", "", "  The   votes are in. ")
	want := "www.example.com news 2026 election results Results The votes are in."
	if got != want {
		t.Errorf("PageQueryText = %q, want %q", got, want)
	}
}
