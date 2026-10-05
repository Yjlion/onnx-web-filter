package classify

import (
	"context"
	"fmt"
	"math"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"

	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/tokenize"
)

// Embedder turns text into sentence embeddings with a catalog site model.
type Embedder struct {
	model    catalog.Model
	spec     catalog.EmbedSpec
	tok      *tokenize.Tokenizer
	sess     *ort.Session
	provider Provider
}

// NewEmbedder opens an installed embedding model and its tokenizer.
func NewEmbedder(m catalog.Model, inst catalog.Installed, o Options) (*Embedder, error) {
	if m.Task != catalog.TaskSite || m.Embed == nil {
		return nil, fmt.Errorf("%s is not an embedding model", m.ID)
	}
	tok, err := tokenize.Load(inst.Path("tokenizer"))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", m.ID, err)
	}
	s, prov, err := openSession(inst.Path("model"), o)
	if err != nil {
		return nil, err
	}
	if err := checkIO(s, m.Embed.Inputs, m.Embed.Output); err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("%s: %w", m.ID, err)
	}
	return &Embedder{model: m, spec: *m.Embed, tok: tok, sess: s, provider: prov}, nil
}

// Model is the catalog entry.
func (e *Embedder) Model() catalog.Model { return e.model }

// Provider says where inference runs.
func (e *Embedder) Provider() Provider { return e.provider }

// Close releases the session.
func (e *Embedder) Close() error { return e.sess.Close() }

// Embed returns one mean-pooled (and, per the catalog, L2-normalised)
// vector per text, running them as one padded batch.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	encs := make([]tokenize.Encoding, len(texts))
	for i, t := range texts {
		encs[i] = e.tok.Encode(t, e.spec.MaxLen)
	}
	inputs, err := tokenInputs(e.spec.Inputs, encs, e.tok)
	if err != nil {
		return nil, err
	}
	hidden, shape, err := runFloat(ctx, e.sess, inputs, e.spec.Output)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", e.model.ID, err)
	}
	if len(shape) != 3 || shape[0] != int64(len(texts)) {
		return nil, fmt.Errorf("%s: unexpected output shape %v", e.model.ID, shape)
	}
	seq, dim := int(shape[1]), int(shape[2])
	out := make([][]float32, len(texts))
	for b, enc := range encs {
		// Mean over the real tokens only: padding rows are excluded, which
		// is what the attention mask means for pooling.
		v := make([]float32, dim)
		n := min(enc.Len(), seq)
		for t := 0; t < n; t++ {
			row := hidden[(b*seq+t)*dim : (b*seq+t+1)*dim]
			for d, x := range row {
				v[d] += x
			}
		}
		for d := range v {
			v[d] /= float32(n)
		}
		if e.spec.Normalize {
			normalize(v)
		}
		out[b] = v
	}
	return out, nil
}

func normalize(v []float32) {
	var s float64
	for _, x := range v {
		s += float64(x) * float64(x)
	}
	if s == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(s))
	for i := range v {
		v[i] *= inv
	}
}

// dot is the cosine similarity of two normalised vectors.
func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}
