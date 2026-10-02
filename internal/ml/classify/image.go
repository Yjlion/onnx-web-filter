package classify

import (
	"bytes"
	"context"
	"fmt"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"

	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
)

// maxDecodePixels refuses to decode images larger than this (40 MP), so a
// small compressed file cannot expand into gigabytes of pixels.
const maxDecodePixels = 40_000_000

// ImageClassifier scores pictures with a catalog image model.
type ImageClassifier struct {
	model    catalog.Model
	spec     catalog.ImageSpec
	weights  []float64
	sess     *ort.Session
	provider Provider
}

// NewImageClassifier opens an installed image model.
func NewImageClassifier(m catalog.Model, inst catalog.Installed, o Options) (*ImageClassifier, error) {
	if m.Task != catalog.TaskImage || m.Image == nil {
		return nil, fmt.Errorf("%s is not an image model", m.ID)
	}
	s, prov, err := openSession(inst.Path("model"), o)
	if err != nil {
		return nil, err
	}
	if err := checkIO(s, []string{m.Image.Input}, m.Image.Output); err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("%s: %w", m.ID, err)
	}
	return &ImageClassifier{model: m, spec: *m.Image, weights: m.UnsafeWeights(), sess: s, provider: prov}, nil
}

// Model is the catalog entry.
func (c *ImageClassifier) Model() catalog.Model { return c.model }

// Provider says where inference runs.
func (c *ImageClassifier) Provider() Provider { return c.provider }

// Close releases the session.
func (c *ImageClassifier) Close() error { return c.sess.Close() }

// Classify decodes JPEG, PNG, GIF (first frame) or WebP bytes and scores
// the picture.
func (c *ImageClassifier) Classify(ctx context.Context, data []byte) (Scores, error) {
	img, err := DecodeImage(data)
	if err != nil {
		return Scores{}, err
	}
	return c.ClassifyImage(ctx, img)
}

// ClassifyImage scores a decoded picture.
func (c *ImageClassifier) ClassifyImage(ctx context.Context, img image.Image) (Scores, error) {
	return c.ClassifyTensor(ctx, Preprocess(img, c.spec))
}

// ClassifyTensor scores an already preprocessed input (Preprocess's
// output); exposed for parity tests against reference runs.
func (c *ImageClassifier) ClassifyTensor(ctx context.Context, x []float32) (Scores, error) {
	n := int64(c.spec.Size)
	shape := []int64{1, 3, n, n}
	if c.spec.Layout == "NHWC" {
		shape = []int64{1, n, n, 3}
	}
	in, err := ort.CreateTensor(shape, x)
	if err != nil {
		return Scores{}, err
	}
	out, _, err := runFloat(ctx, c.sess, map[string]*ort.Tensor{c.spec.Input: in}, c.spec.Output)
	if err != nil {
		return Scores{}, fmt.Errorf("%s: %w", c.model.ID, err)
	}
	return scores(out, c.spec.Softmax, c.model.Labels, c.weights)
}

// DecodeImage decodes image bytes, refusing decompression bombs.
func DecodeImage(data []byte) (image.Image, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > maxDecodePixels {
		return nil, fmt.Errorf("decode image: %dx%d is out of range", cfg.Width, cfg.Height)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return img, nil
}

// Preprocess resizes img to the model's square input (bilinear, aspect
// ratio not kept, as the models were trained), composites transparency
// onto white, scales to [0,1], normalises with the spec's mean and std, and
// lays the pixels out as NCHW or NHWC.
func Preprocess(img image.Image, spec catalog.ImageSpec) []float32 {
	n := spec.Size
	dst := image.NewRGBA(image.Rect(0, 0, n, n))
	draw.Draw(dst, dst.Bounds(), &image.Uniform{C: color.White}, image.Point{}, draw.Src)
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Over, nil)

	out := make([]float32, 3*n*n)
	plane := n * n
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			off := dst.PixOffset(x, y)
			for ch := 0; ch < 3; ch++ {
				v := (float32(dst.Pix[off+ch])/255 - spec.Mean[ch]) / spec.Std[ch]
				if spec.Layout == "NHWC" {
					out[(y*n+x)*3+ch] = v
				} else {
					out[ch*plane+y*n+x] = v
				}
			}
		}
	}
	return out
}
