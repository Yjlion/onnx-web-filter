package classify

import (
	"context"
	"fmt"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"

	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/tokenize"
)

// TextClassifier scores page text with a catalog sequence classifier.
type TextClassifier struct {
	model    catalog.Model
	spec     catalog.TextSpec
	weights  []float64
	tok      *tokenize.Tokenizer
	sess     *ort.Session
	provider Provider
}

// NewTextClassifier opens an installed text model and its tokenizer.
func NewTextClassifier(m catalog.Model, inst catalog.Installed, o Options) (*TextClassifier, error) {
	if m.Task != catalog.TaskText || m.Text == nil {
		return nil, fmt.Errorf("%s is not a text model", m.ID)
	}
	tok, err := tokenize.Load(inst.Path("tokenizer"))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", m.ID, err)
	}
	s, prov, err := openSession(inst.Path("model"), o)
	if err != nil {
		return nil, err
	}
	if err := checkIO(s, m.Text.Inputs, m.Text.Output); err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("%s: %w", m.ID, err)
	}
	return &TextClassifier{model: m, spec: *m.Text, weights: m.UnsafeWeights(), tok: tok, sess: s, provider: prov}, nil
}

// Model is the catalog entry.
func (c *TextClassifier) Model() catalog.Model { return c.model }

// Provider says where inference runs.
func (c *TextClassifier) Provider() Provider { return c.provider }

// Close releases the session.
func (c *TextClassifier) Close() error { return c.sess.Close() }

// Classify scores text; anything past the model's max_len tokens is cut.
func (c *TextClassifier) Classify(ctx context.Context, text string) (Scores, error) {
	enc := c.tok.Encode(text, c.spec.MaxLen)
	inputs, err := tokenInputs(c.spec.Inputs, []tokenize.Encoding{enc}, c.tok)
	if err != nil {
		return Scores{}, err
	}
	out, _, err := runFloat(ctx, c.sess, inputs, c.spec.Output)
	if err != nil {
		return Scores{}, fmt.Errorf("%s: %w", c.model.ID, err)
	}
	return scores(out, c.spec.Softmax, c.model.Labels, c.weights)
}

// tokenInputs builds the named [batch, seq] int64 tensors for a padded
// batch. On error, tensors already made are closed.
func tokenInputs(names []string, encs []tokenize.Encoding, tok *tokenize.Tokenizer) (map[string]*ort.Tensor, error) {
	ids, mask, types, seqLen := tok.Pad(encs)
	shape := []int64{int64(len(encs)), int64(seqLen)}
	out := make(map[string]*ort.Tensor, len(names))
	for _, name := range names {
		var data []int64
		switch name {
		case "input_ids":
			data = ids
		case "attention_mask":
			data = mask
		case "token_type_ids":
			data = types
		default:
			for _, t := range out {
				_ = t.Close()
			}
			return nil, fmt.Errorf("unsupported model input %q", name)
		}
		t, err := ort.CreateTensor(shape, data)
		if err != nil {
			for _, t := range out {
				_ = t.Close()
			}
			return nil, err
		}
		out[name] = t
	}
	return out, nil
}
