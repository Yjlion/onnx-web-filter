package classify

import (
	"context"
	"fmt"
	"math"
	"net/url"
	"sort"
	"strings"
	"unicode"
)

// Prototype is one category and the texts that describe it.
type Prototype struct {
	Slug  string
	Texts []string
}

// SiteClassifier sorts sites into categories without training: it embeds
// "host title description" and compares it with the embedded prototype
// texts of every category. A category scores its best-matching prototype;
// a softmax over the category scores gives the confidence.
type SiteClassifier struct {
	emb         *Embedder
	slugs       []string
	vecs        [][]float32 // prototype vectors
	owner       []int       // vecs[i] belongs to slugs[owner[i]]
	temperature float64
}

// DefaultTemperature sharpens the softmax over cosine similarities, which
// for this kind of model sit in a narrow band (0.2-0.7). Tuned on
// testdata/sites.tsv.
const DefaultTemperature = 0.05

// NewSiteClassifier embeds the prototypes. Each text is embedded on its own,
// as queries are, so prototype vectors do not depend on batching.
func NewSiteClassifier(ctx context.Context, emb *Embedder, protos []Prototype, temperature float64) (*SiteClassifier, error) {
	if temperature <= 0 {
		temperature = DefaultTemperature
	}
	c := &SiteClassifier{emb: emb, temperature: temperature}
	for _, p := range protos {
		if len(p.Texts) == 0 {
			return nil, fmt.Errorf("category %s has no prototype texts", p.Slug)
		}
		idx := len(c.slugs)
		c.slugs = append(c.slugs, p.Slug)
		for _, t := range p.Texts {
			v, err := emb.Embed(ctx, []string{t})
			if err != nil {
				return nil, err
			}
			c.vecs = append(c.vecs, v[0])
			c.owner = append(c.owner, idx)
		}
	}
	if len(c.slugs) < 2 {
		return nil, fmt.Errorf("need at least two categories, have %d", len(c.slugs))
	}
	return c, nil
}

// SlugScore is one category's share of the confidence.
type SlugScore struct {
	Slug        string  `json:"slug"`
	Similarity  float64 `json:"similarity"` // best prototype cosine
	Probability float64 `json:"probability"`
}

// SiteScores is the answer for one site, best category first.
type SiteScores struct {
	Category   string      `json:"category"`
	Confidence float64     `json:"confidence"`
	Ranked     []SlugScore `json:"ranked"`
}

// QueryText is what a site looks like to the embedder.
func QueryText(host, title, description string) string {
	parts := []string{strings.TrimSpace(host)}
	if t := strings.TrimSpace(title); t != "" {
		parts = append(parts, t)
	}
	if d := strings.TrimSpace(description); d != "" {
		if r := []rune(d); len(r) > 300 {
			d = string(r[:300])
		}
		parts = append(parts, d)
	}
	return strings.Join(parts, " ")
}

// PageQueryText is what one page looks like to the embedder: its site's
// query text with the URL path's words after the host, then the start of
// the page's visible text. The host, title and description lead so the
// category prototypes still match; the model reads 256 tokens.
func PageQueryText(rawURL, title, description, text string) string {
	host := rawURL
	if u, err := url.Parse(rawURL); err == nil && u.Host != "" {
		host = u.Hostname()
		if words := strings.Join(strings.FieldsFunc(u.Path, func(r rune) bool {
			return !unicode.IsLetter(r) && !unicode.IsDigit(r)
		}), " "); words != "" {
			host += " " + words
		}
	}
	q := QueryText(host, title, description)
	if t := strings.Join(strings.Fields(text), " "); t != "" {
		if r := []rune(t); len(r) > 600 {
			t = string(r[:600])
		}
		q += " " + t
	}
	return q
}

// Classify ranks the categories for a site. title and description may be
// empty; the host alone is usually enough for well-known sites.
func (c *SiteClassifier) Classify(ctx context.Context, host, title, description string) (SiteScores, error) {
	return c.ClassifyQuery(ctx, QueryText(host, title, description))
}

// ClassifyPage ranks the categories for one page from its URL and content.
func (c *SiteClassifier) ClassifyPage(ctx context.Context, rawURL, title, description, text string) (SiteScores, error) {
	return c.ClassifyQuery(ctx, PageQueryText(rawURL, title, description, text))
}

// ClassifyQuery ranks the categories for an already assembled query text.
func (c *SiteClassifier) ClassifyQuery(ctx context.Context, query string) (SiteScores, error) {
	v, err := c.emb.Embed(ctx, []string{query})
	if err != nil {
		return SiteScores{}, err
	}
	best := make([]float64, len(c.slugs))
	for i := range best {
		best[i] = math.Inf(-1)
	}
	for i, pv := range c.vecs {
		best[c.owner[i]] = max(best[c.owner[i]], dot(v[0], pv))
	}
	logits := make([]float64, len(best))
	for i, s := range best {
		logits[i] = s / c.temperature
	}
	probs := softmaxF(logits)
	ranked := make([]SlugScore, len(c.slugs))
	for i, slug := range c.slugs {
		ranked[i] = SlugScore{Slug: slug, Similarity: best[i], Probability: probs[i]}
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].Probability > ranked[j].Probability })
	return SiteScores{Category: ranked[0].Slug, Confidence: ranked[0].Probability, Ranked: ranked[:min(5, len(ranked))]}, nil
}
