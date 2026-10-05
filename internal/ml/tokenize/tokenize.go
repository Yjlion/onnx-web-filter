// Package tokenize turns text into model input ids, reading the Hugging Face
// tokenizer.json a model ships with. It implements the pieces BERT-family
// classifiers and sentence embedders use (BertNormalizer, BertPreTokenizer,
// WordPiece, a [CLS] A [SEP] template), matching the Rust `tokenizers`
// library token for token; testdata/golden.json is produced by it (see
// testdata/gen_golden.py).
//
// One deliberate difference: special tokens written in the input ("[SEP]"
// in page text) are tokenized as ordinary text, not matched as the special
// token, so a page cannot inject sequence structure.
package tokenize

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer encodes text for one model. It is safe for concurrent use.
type Tokenizer struct {
	vocab map[string]int64

	lowercase, stripAccents, cleanText, chineseChars bool

	unkID       int64
	prefix      string // continuing subword prefix, "##"
	maxWordLen  int    // max_input_chars_per_word
	clsID       int64
	sepID       int64
	padID       int64
	maxLen      int // from tokenizer.json truncation; 0 = none
	vocabLength int
}

// Encoding is one encoded sequence, special tokens included.
type Encoding struct {
	IDs           []int64
	AttentionMask []int64
	TypeIDs       []int64
}

// Len is the number of tokens.
func (e Encoding) Len() int { return len(e.IDs) }

// tokenizerJSON is the subset of tokenizer.json this package reads.
type tokenizerJSON struct {
	Normalizer *struct {
		Type         string `json:"type"`
		CleanText    bool   `json:"clean_text"`
		ChineseChars bool   `json:"handle_chinese_chars"`
		StripAccents *bool  `json:"strip_accents"`
		Lowercase    bool   `json:"lowercase"`
	} `json:"normalizer"`
	PreTokenizer *struct {
		Type string `json:"type"`
	} `json:"pre_tokenizer"`
	PostProcessor *struct {
		Type   string `json:"type"`
		Single []struct {
			SpecialToken *struct {
				ID string `json:"id"`
			} `json:"SpecialToken"`
			Sequence *struct {
				ID string `json:"id"`
			} `json:"Sequence"`
		} `json:"single"`
		// BertProcessing form.
		CLS []any `json:"cls"`
		SEP []any `json:"sep"`
	} `json:"post_processor"`
	Truncation *struct {
		MaxLength int `json:"max_length"`
	} `json:"truncation"`
	Padding *struct {
		PadID int64 `json:"pad_id"`
	} `json:"padding"`
	Model struct {
		Type          string           `json:"type"`
		UnkToken      string           `json:"unk_token"`
		Prefix        string           `json:"continuing_subword_prefix"`
		MaxInputChars int              `json:"max_input_chars_per_word"`
		Vocab         map[string]int64 `json:"vocab"`
	} `json:"model"`
}

// Load reads a tokenizer.json file.
func Load(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t, err := Parse(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

// Parse reads tokenizer.json content. Anything other than the BERT
// pipeline is rejected rather than tokenized wrongly.
func Parse(data []byte) (*Tokenizer, error) {
	var j tokenizerJSON
	if err := json.Unmarshal(data, &j); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}
	if j.Model.Type != "WordPiece" {
		return nil, fmt.Errorf("unsupported tokenizer model %q (only WordPiece)", j.Model.Type)
	}
	if len(j.Model.Vocab) == 0 {
		return nil, fmt.Errorf("tokenizer.json has an empty vocab")
	}
	t := &Tokenizer{
		vocab:       j.Model.Vocab,
		prefix:      j.Model.Prefix,
		maxWordLen:  j.Model.MaxInputChars,
		vocabLength: len(j.Model.Vocab),
	}
	if t.maxWordLen <= 0 {
		t.maxWordLen = 100
	}
	if j.Normalizer != nil {
		if j.Normalizer.Type != "BertNormalizer" {
			return nil, fmt.Errorf("unsupported normalizer %q (only BertNormalizer)", j.Normalizer.Type)
		}
		t.cleanText = j.Normalizer.CleanText
		t.chineseChars = j.Normalizer.ChineseChars
		t.lowercase = j.Normalizer.Lowercase
		// strip_accents unset follows lowercase, as in the Rust library.
		t.stripAccents = t.lowercase
		if j.Normalizer.StripAccents != nil {
			t.stripAccents = *j.Normalizer.StripAccents
		}
	}
	if j.PreTokenizer != nil && j.PreTokenizer.Type != "BertPreTokenizer" {
		return nil, fmt.Errorf("unsupported pre-tokenizer %q (only BertPreTokenizer)", j.PreTokenizer.Type)
	}
	var ok bool
	if t.unkID, ok = t.vocab[j.Model.UnkToken]; !ok {
		return nil, fmt.Errorf("unk token %q not in vocab", j.Model.UnkToken)
	}
	cls, sep := "[CLS]", "[SEP]"
	if pp := j.PostProcessor; pp != nil {
		switch pp.Type {
		case "TemplateProcessing":
			// Expect exactly: SpecialToken, Sequence A, SpecialToken.
			if len(pp.Single) != 3 || pp.Single[0].SpecialToken == nil || pp.Single[1].Sequence == nil || pp.Single[2].SpecialToken == nil {
				return nil, fmt.Errorf("unsupported post-processor template (want <special> $A <special>)")
			}
			cls, sep = pp.Single[0].SpecialToken.ID, pp.Single[2].SpecialToken.ID
		case "BertProcessing":
			if len(pp.CLS) == 2 && len(pp.SEP) == 2 {
				cls, _ = pp.CLS[0].(string)
				sep, _ = pp.SEP[0].(string)
			}
		default:
			return nil, fmt.Errorf("unsupported post-processor %q", pp.Type)
		}
	}
	if t.clsID, ok = t.vocab[cls]; !ok {
		return nil, fmt.Errorf("token %q not in vocab", cls)
	}
	if t.sepID, ok = t.vocab[sep]; !ok {
		return nil, fmt.Errorf("token %q not in vocab", sep)
	}
	if id, ok := t.vocab["[PAD]"]; ok {
		t.padID = id
	}
	if j.Padding != nil {
		t.padID = j.Padding.PadID
	}
	if j.Truncation != nil {
		t.maxLen = j.Truncation.MaxLength
	}
	return t, nil
}

// MaxLen is the truncation length tokenizer.json sets, or 0.
func (t *Tokenizer) MaxLen() int { return t.maxLen }

// PadID is the id used to pad batches.
func (t *Tokenizer) PadID() int64 { return t.padID }

// VocabSize is the number of vocab entries.
func (t *Tokenizer) VocabSize() int { return t.vocabLength }

// Encode tokenizes text as [CLS] text [SEP], truncating the text so the
// whole sequence is at most maxLen tokens. maxLen 0 uses the tokenizer.json
// truncation length (none if it has none); a negative maxLen disables
// truncation. A maxLen below 2 is raised to 2, which leaves no room for text.
func (t *Tokenizer) Encode(text string, maxLen int) Encoding {
	switch {
	case maxLen == 0:
		maxLen = t.maxLen
	case maxLen < 0:
		maxLen = 0
	case maxLen < 2:
		maxLen = 2
	}
	ids := []int64{t.clsID}
	for _, word := range t.preTokenize(t.normalize(text)) {
		if maxLen > 0 && len(ids) >= maxLen-1 {
			break
		}
		ids = t.wordpiece(ids, word)
	}
	if maxLen > 0 && len(ids) > maxLen-1 {
		ids = ids[:maxLen-1]
	}
	ids = append(ids, t.sepID)
	mask := make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return Encoding{IDs: ids, AttentionMask: mask, TypeIDs: make([]int64, len(ids))}
}

// Tokens returns the word pieces of text without special tokens or
// truncation; for debugging and tests.
func (t *Tokenizer) Tokens(text string) []string {
	inv := make(map[int64]string, len(t.vocab))
	for k, v := range t.vocab {
		inv[v] = k
	}
	var out []string
	for _, word := range t.preTokenize(t.normalize(text)) {
		for _, id := range t.wordpiece(nil, word) {
			out = append(out, inv[id])
		}
	}
	return out
}

// Pad right-pads encodings to the longest one, for a batched run. The
// returned slices are flattened row-major [batch, seqLen].
func (t *Tokenizer) Pad(encs []Encoding) (ids, mask, typeIDs []int64, seqLen int) {
	for _, e := range encs {
		seqLen = max(seqLen, e.Len())
	}
	n := len(encs) * seqLen
	ids, mask, typeIDs = make([]int64, n), make([]int64, n), make([]int64, n)
	for i, e := range encs {
		row := i * seqLen
		copy(ids[row:], e.IDs)
		copy(mask[row:], e.AttentionMask)
		copy(typeIDs[row:], e.TypeIDs)
		for k := row + e.Len(); k < row+seqLen; k++ {
			ids[k] = t.padID
		}
	}
	return ids, mask, typeIDs, seqLen
}

// normalize is BertNormalizer: clean text, pad CJK ideographs with spaces,
// strip accents, lowercase - in that order, as the Rust library does.
func (t *Tokenizer) normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if t.cleanText {
			if r == 0 || r == 0xFFFD || isControl(r) {
				continue
			}
			if isWhitespace(r) {
				r = ' '
			}
		}
		if t.chineseChars && isChinese(r) {
			b.WriteByte(' ')
			b.WriteRune(r)
			b.WriteByte(' ')
			continue
		}
		b.WriteRune(r)
	}
	s = b.String()
	if t.stripAccents {
		b.Reset()
		for _, r := range norm.NFD.String(s) {
			if !unicode.Is(unicode.Mn, r) {
				b.WriteRune(r)
			}
		}
		s = b.String()
	}
	if t.lowercase {
		b.Reset()
		for _, r := range s {
			// Rust's char::to_lowercase maps U+0130 to two chars; every
			// other mapping is one to one, which unicode.ToLower covers.
			if r == 'İ' {
				b.WriteString("i̇")
				continue
			}
			b.WriteRune(unicode.ToLower(r))
		}
		s = b.String()
	}
	return s
}

// preTokenize is BertPreTokenizer: split on whitespace, and make every
// punctuation character a word of its own.
func (t *Tokenizer) preTokenize(s string) []string {
	var words []string
	start := -1
	for i, r := range s {
		switch {
		case isWhitespace(r):
			if start >= 0 {
				words = append(words, s[start:i])
				start = -1
			}
		case isPunctuation(r):
			if start >= 0 {
				words = append(words, s[start:i])
				start = -1
			}
			words = append(words, string(r))
		default:
			if start < 0 {
				start = i
			}
		}
	}
	if start >= 0 {
		words = append(words, s[start:])
	}
	return words
}

// wordpiece appends word's pieces to ids: greedy longest-match-first
// against the vocab, continuation pieces carrying the prefix. A word that
// cannot be split, or is longer than maxWordLen characters, becomes [UNK].
func (t *Tokenizer) wordpiece(ids []int64, word string) []int64 {
	runes := []rune(word)
	if len(runes) > t.maxWordLen {
		return append(ids, t.unkID)
	}
	mark := len(ids)
	var sb strings.Builder
	for start := 0; start < len(runes); {
		end := len(runes)
		found := int64(-1)
		for ; end > start; end-- {
			sb.Reset()
			if start > 0 {
				sb.WriteString(t.prefix)
			}
			sb.WriteString(string(runes[start:end]))
			if id, ok := t.vocab[sb.String()]; ok {
				found = id
				break
			}
		}
		if found < 0 {
			return append(ids[:mark], t.unkID)
		}
		ids = append(ids, found)
		start = end
	}
	return ids
}

// isWhitespace is Rust's char::is_whitespace (the White_Space property).
func isWhitespace(r rune) bool {
	return unicode.Is(unicode.White_Space, r)
}

// isControl matches the Rust library: tab, newline and carriage return are
// whitespace, anything else in the "Other" categories (Cc, Cf, Co, Cs, Cn)
// is a control character.
func isControl(r rune) bool {
	switch r {
	case '\t', '\n', '\r':
		return false
	}
	return unicode.In(r, unicode.Cc, unicode.Cf, unicode.Co, unicode.Cs) || isUnassigned(r)
}

func isUnassigned(r rune) bool {
	return !unicode.In(r, unicode.L, unicode.M, unicode.N, unicode.P, unicode.S, unicode.Z, unicode.C)
}

// isPunctuation is ASCII punctuation (which includes symbols such as $ + ^
// that Unicode files under S*) or any Unicode P* character.
func isPunctuation(r rune) bool {
	if r < 0x80 {
		return (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126)
	}
	return unicode.IsPunct(r)
}

// isChinese uses the Rust library's CJK ranges (which differ from the
// original BERT code in the Extension E start, 0x2B920).
func isChinese(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B920 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}
