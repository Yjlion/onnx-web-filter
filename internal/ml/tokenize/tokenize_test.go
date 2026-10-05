package tokenize

import (
	"encoding/json"
	"os"
	"slices"
	"strings"
	"testing"
)

type goldenCase struct {
	Text   string   `json:"text"`
	MaxLen int      `json:"max_len"`
	IDs    []int64  `json:"ids"`
	Tokens []string `json:"tokens"`
}

func load(t testing.TB) *Tokenizer {
	t.Helper()
	tok, err := Load("testdata/tokenizer.json")
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestGolden compares against the Rust tokenizers library's output for the
// same tokenizer.json (testdata/gen_golden.py).
func TestGolden(t *testing.T) {
	tok := load(t)
	data, err := os.ReadFile("testdata/golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var cases []goldenCase
	if err := json.Unmarshal(data, &cases); err != nil {
		t.Fatal(err)
	}
	for _, c := range cases {
		enc := tok.Encode(c.Text, c.MaxLen)
		if !slices.Equal(enc.IDs, c.IDs) {
			name := c.Text
			if len(name) > 40 {
				name = name[:40] + "..."
			}
			t.Errorf("%q (max %d):\n got  %v\n want %v (%v)\n go tokens %v", name, c.MaxLen, enc.IDs, c.IDs, c.Tokens, tok.Tokens(c.Text))
			continue
		}
		if len(enc.AttentionMask) != len(enc.IDs) || len(enc.TypeIDs) != len(enc.IDs) {
			t.Errorf("mask/type lengths differ from ids for %q", c.Text)
		}
	}
}

func TestDefaultsFromTokenizerJSON(t *testing.T) {
	tok := load(t)
	if tok.MaxLen() != 128 || tok.PadID() != 0 || tok.VocabSize() != 30522 {
		t.Errorf("maxLen=%d pad=%d vocab=%d", tok.MaxLen(), tok.PadID(), tok.VocabSize())
	}
	long := strings.Repeat("word ", 500)
	if n := tok.Encode(long, 0).Len(); n != 128 {
		t.Errorf("default truncation: %d tokens, want 128", n)
	}
	if n := tok.Encode(long, -1).Len(); n != 502 {
		t.Errorf("no truncation: %d tokens, want 502", n)
	}
	if enc := tok.Encode(long, 1); !slices.Equal(enc.IDs, []int64{101, 102}) {
		t.Errorf("maxLen 1 should still give [CLS] [SEP], got %v", enc.IDs)
	}
}

func TestSpecialTokensInTextAreNotInjected(t *testing.T) {
	tok := load(t)
	enc := tok.Encode("safe [SEP] text [CLS]", -1)
	for _, id := range enc.IDs[1 : len(enc.IDs)-1] {
		if id == 101 || id == 102 {
			t.Fatalf("special token injected from input text: %v", enc.IDs)
		}
	}
}

func TestPad(t *testing.T) {
	tok := load(t)
	a, b := tok.Encode("short", -1), tok.Encode("a somewhat longer sentence", -1)
	ids, mask, types, n := tok.Pad([]Encoding{a, b})
	if n != b.Len() || len(ids) != 2*n || len(mask) != 2*n || len(types) != 2*n {
		t.Fatalf("seqLen=%d lens=%d/%d/%d", n, len(ids), len(mask), len(types))
	}
	if !slices.Equal(ids[:a.Len()], a.IDs) || ids[a.Len()] != tok.PadID() || mask[a.Len()] != 0 || mask[a.Len()-1] != 1 {
		t.Errorf("row 0 not padded correctly: ids=%v mask=%v", ids[:n], mask[:n])
	}
	if !slices.Equal(ids[n:], b.IDs) {
		t.Errorf("row 1 = %v, want %v", ids[n:], b.IDs)
	}
}

func TestRejectsUnsupportedPipelines(t *testing.T) {
	for _, bad := range []string{
		`{"model":{"type":"Unigram","vocab":{"a":0}}}`,
		`{"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{}}}`,
		`{"normalizer":{"type":"NFKC"},"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{"[UNK]":0}}}`,
		`{"model":{"type":"WordPiece","unk_token":"[UNK]","vocab":{"x":0}}}`,
	} {
		if _, err := Parse([]byte(bad)); err == nil {
			t.Errorf("Parse accepted %s", bad)
		}
	}
}

func BenchmarkEncodePage(b *testing.B) {
	tok := load(b)
	text := strings.Repeat("The quick brown fox jumps over the lazy dog, again and again. ", 60)
	b.ReportAllocs()
	for b.Loop() {
		tok.Encode(text, 512)
	}
}
