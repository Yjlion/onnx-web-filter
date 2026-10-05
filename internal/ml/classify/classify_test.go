package classify

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortenv"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
)

// Tests against real models need a data dir holding the runtime and the
// models (`webfilter ml download --data-dir DIR`):
//
//	ML_DATA_DIR=$PWD/data/ml go test ./internal/ml/classify
//
// ORT_LIB_PATH, if set, overrides the runtime in that dir.
func mlDataDir(t testing.TB) string {
	t.Helper()
	dir := os.Getenv("ML_DATA_DIR")
	if dir == "" {
		t.Skip("ML_DATA_DIR not set")
	}
	initOnce.Do(func() {
		lib, err := ortenv.Locate("", dir, ortrt.AccelCPU)
		if err == nil {
			_, err = ortenv.Init(lib)
		}
		initErr = err
	})
	if initErr != nil {
		t.Fatal(initErr)
	}
	return dir
}

var (
	initOnce sync.Once
	initErr  error
)

func installed(t testing.TB, id string) (catalog.Model, catalog.Installed) {
	t.Helper()
	dir := mlDataDir(t)
	m, ok := catalog.Lookup(id)
	if !ok {
		t.Fatalf("no catalog model %s", id)
	}
	inst, ok := catalog.LoadInstalled(dir, m)
	if !ok {
		t.Skipf("%s not installed in %s", id, dir)
	}
	return m, inst
}

type fixtures struct {
	Images []struct {
		Model string    `json:"model"`
		Case  string    `json:"case"`
		Seed  int64     `json:"seed"`
		Probs []float64 `json:"probs"`
	} `json:"images"`
	Texts []struct {
		Model string    `json:"model"`
		Text  string    `json:"text"`
		Probs []float64 `json:"probs"`
	} `json:"texts"`
	Embeds []struct {
		Model  string    `json:"model"`
		Text   string    `json:"text"`
		Vector []float64 `json:"vector"`
	} `json:"embeds"`
}

func loadFixtures(t testing.TB) fixtures {
	t.Helper()
	data, err := os.ReadFile("testdata/fixtures.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx fixtures
	if err := json.Unmarshal(data, &fx); err != nil {
		t.Fatal(err)
	}
	return fx
}

// lcg reproduces gen_fixtures.py's deterministic tensor.
func lcg(seed int64, n int) []float32 {
	out := make([]float32, n)
	s := uint32(seed)
	for i := range out {
		s = 1664525*s + 1013904223
		out[i] = float32(float64(s) / 4294967296.0)
	}
	return out
}

func maxDiff(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var d float64
	for i := range a {
		d = max(d, math.Abs(a[i]-b[i]))
	}
	return d
}

func TestImageModelsMatchReference(t *testing.T) {
	fx := loadFixtures(t)
	ctx := context.Background()
	for _, id := range []string{"nsfwjs-mobilenet", "falconsai-vit"} {
		t.Run(id, func(t *testing.T) {
			m, inst := installed(t, id)
			c, err := NewImageClassifier(m, inst, Options{})
			if err != nil {
				t.Fatal(err)
			}
			defer c.Close()
			n := 3 * m.Image.Size * m.Image.Size
			for _, f := range fx.Images {
				if f.Model != id {
					continue
				}
				var got Scores
				switch f.Seed {
				case -1:
					x := make([]float32, n)
					for i := range x {
						x[i] = 0.5
					}
					got, err = c.ClassifyTensor(ctx, x)
				case -2:
					// Go's bilinear resampler differs a little from PIL's.
					data, _ := os.ReadFile("testdata/scene.jpg")
					got, err = c.Classify(ctx, data)
				default:
					got, err = c.ClassifyTensor(ctx, lcg(f.Seed, n))
				}
				if err != nil {
					t.Fatalf("%s: %v", f.Case, err)
				}
				// The fixtures come from one CPU; ONNX Runtime picks different
				// int8 kernels by instruction set (AVX2, AVX-512, VNNI, NEON),
				// which moves quantized outputs by up to ~1e-3.
				tol := 2e-3
				if f.Seed == -2 {
					tol = 0.05
				}
				if d := maxDiff(got.Probs, f.Probs); d > tol {
					t.Errorf("%s: probs %v, reference %v (diff %.5f > %g)", f.Case, got.Probs, f.Probs, d, tol)
				}
			}
		})
	}
}

func TestImageScoresAndTransparency(t *testing.T) {
	m, inst := installed(t, "nsfwjs-mobilenet")
	c, err := NewImageClassifier(m, inst, Options{Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if c.Provider() != "cpu" {
		t.Errorf("provider %q", c.Provider())
	}
	// A fully transparent PNG is composited onto white: it must classify
	// like a white square, not like black.
	img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	transparent, err := c.Classify(context.Background(), buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	white := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	for i := range white.Pix {
		white.Pix[i] = 255
	}
	ws, _ := c.ClassifyImage(context.Background(), white)
	if maxDiff(transparent.Probs, ws.Probs) > 1e-6 {
		t.Errorf("transparent %v != white %v", transparent.Probs, ws.Probs)
	}
	var sum float64
	for _, p := range ws.Probs {
		sum += p
	}
	if math.Abs(sum-1) > 1e-3 || ws.Unsafe < 0 || ws.Unsafe > 1 {
		t.Errorf("probs %v sum %.4f unsafe %.4f", ws.Probs, sum, ws.Unsafe)
	}
	if _, err := c.Classify(context.Background(), []byte("not an image")); err == nil {
		t.Error("garbage bytes classified")
	}
}

func TestImageConcurrent(t *testing.T) {
	m, inst := installed(t, "nsfwjs-mobilenet")
	c, err := NewImageClassifier(m, inst, Options{Threads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	img := image.NewRGBA(image.Rect(0, 0, 100, 80))
	for i := range img.Pix {
		img.Pix[i] = uint8(i * 7)
	}
	want, _ := c.ClassifyImage(context.Background(), img)
	var wg sync.WaitGroup
	for range 8 {
		wg.Go(func() {
			for range 5 {
				got, err := c.ClassifyImage(context.Background(), img)
				if err != nil || maxDiff(got.Probs, want.Probs) > 1e-6 {
					t.Errorf("concurrent run: %v %v", got.Probs, err)
				}
			}
		})
	}
	wg.Wait()
}

func TestDecodeRejectsBombs(t *testing.T) {
	// A PNG header claiming 100000x100000 pixels.
	img := image.NewGray(image.Rect(0, 0, 1, 1))
	img.Set(0, 0, color.Gray{})
	var buf bytes.Buffer
	_ = png.Encode(&buf, img)
	b := buf.Bytes()
	// IHDR width/height live at bytes 16..24. Whether DecodeConfig then
	// reports the huge size or trips over the stale CRC, DecodeImage must
	// refuse rather than decode.
	b[16], b[17], b[18], b[19] = 0, 1, 0x86, 0xA0
	b[20], b[21], b[22], b[23] = 0, 1, 0x86, 0xA0
	if _, err := DecodeImage(b); err == nil {
		t.Fatal("oversized image decoded")
	}
}

func TestPreprocessLayout(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 255, 0, 51, 255
	}
	nchw := Preprocess(img, catalog.ImageSpec{Size: 2, Layout: "NCHW", Mean: [3]float32{0.5, 0.5, 0.5}, Std: [3]float32{0.5, 0.5, 0.5}})
	nhwc := Preprocess(img, catalog.ImageSpec{Size: 2, Layout: "NHWC", Std: [3]float32{1, 1, 1}})
	near := func(a, b float32) bool { return math.Abs(float64(a-b)) < 1e-5 }
	// NCHW: 4 reds, then 4 greens, then 4 blues; normalised to [-1,1].
	if !near(nchw[0], 1) || !near(nchw[4], -1) || !near(nchw[8], 0.2*2-1) {
		t.Errorf("NCHW = %v", nchw)
	}
	// NHWC: r,g,b per pixel in [0,1].
	if !near(nhwc[0], 1) || !near(nhwc[1], 0) || !near(nhwc[2], 0.2) || !near(nhwc[3], 1) {
		t.Errorf("NHWC = %v", nhwc)
	}
}

func BenchmarkImageNSFWJS(b *testing.B) { benchImage(b, "nsfwjs-mobilenet") }
func BenchmarkImageViT(b *testing.B)    { benchImage(b, "falconsai-vit") }

func benchImage(b *testing.B, id string) {
	m, inst := installed(b, id)
	c, err := NewImageClassifier(m, inst, Options{})
	if err != nil {
		b.Fatal(err)
	}
	defer c.Close()
	data, _ := os.ReadFile("testdata/scene.jpg")
	b.ResetTimer()
	for b.Loop() {
		if _, err := c.Classify(context.Background(), data); err != nil {
			b.Fatal(err)
		}
	}
}

func TestTextModelMatchesReference(t *testing.T) {
	fx := loadFixtures(t)
	m, inst := installed(t, "distilbert-nsfw")
	c, err := NewTextClassifier(m, inst, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if len(fx.Texts) == 0 {
		t.Fatal("no text fixtures")
	}
	for _, f := range fx.Texts {
		got, err := c.Classify(context.Background(), f.Text)
		if err != nil {
			t.Fatal(err)
		}
		// Same CPU-dependent int8 drift as the image models.
		if d := maxDiff(got.Probs, f.Probs); d > 2e-3 {
			t.Errorf("%.40q: probs %v, reference %v", f.Text, got.Probs, f.Probs)
		}
		// The weighted score is p(nsfw) for this two-label model.
		if math.Abs(got.Unsafe-got.Probs[1]) > 1e-9 {
			t.Errorf("unsafe %.4f, want %.4f", got.Unsafe, f.Probs[1])
		}
	}
}

func TestTextLongAndEmpty(t *testing.T) {
	m, inst := installed(t, "distilbert-nsfw")
	c, err := NewTextClassifier(m, inst, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	long := strings.Repeat("The museum opens a new exhibition about medieval farming tools. ", 400)
	if _, err := c.Classify(context.Background(), long); err != nil {
		t.Fatalf("long text: %v", err)
	}
	if _, err := c.Classify(context.Background(), ""); err != nil {
		t.Fatalf("empty text: %v", err)
	}
}

func BenchmarkTextPage(b *testing.B) {
	m, inst := installed(b, "distilbert-nsfw")
	c, err := NewTextClassifier(m, inst, Options{})
	if err != nil {
		b.Fatal(err)
	}
	defer c.Close()
	page := strings.Repeat("The museum opens a new exhibition about medieval farming tools. ", 60)
	for b.Loop() {
		if _, err := c.Classify(context.Background(), page); err != nil {
			b.Fatal(err)
		}
	}
}

func TestEmbedderMatchesReference(t *testing.T) {
	fx := loadFixtures(t)
	m, inst := installed(t, "minilm-l6")
	e, err := NewEmbedder(m, inst, Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer e.Close()
	var texts []string
	for _, f := range fx.Embeds {
		texts = append(texts, f.Text)
	}
	if len(texts) == 0 {
		t.Fatal("no embedding fixtures")
	}
	// The int8 model quantizes activations with one scale per tensor, so
	// a text's vector depends slightly on its batch-mates: compare single
	// runs exactly and a batch loosely.
	batch, err := e.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	vecs := make([][]float32, len(texts))
	for i, f := range fx.Embeds {
		v, err := e.Embed(context.Background(), []string{f.Text})
		if err != nil {
			t.Fatal(err)
		}
		vecs[i] = v[0]
		ref := make([]float32, len(f.Vector))
		for d, x := range f.Vector {
			ref[d] = float32(x)
		}
		// Compared by direction, which is what the site classifier uses:
		// int8 kernels differ between CPUs and move single components by a
		// few hundredths. Unrelated texts sit below 0.7, so 0.95 still
		// catches broken tokenizing or pooling.
		if c := dot(v[0], ref); c < 0.95 {
			t.Errorf("%q: cosine to reference %.4f", f.Text, c)
		}
		if c := dot(batch[i], v[0]); c < 0.99 {
			t.Errorf("%q: batched vs single cosine %.4f", f.Text, c)
		}
		if n := dot(vecs[i], vecs[i]); math.Abs(n-1) > 1e-4 {
			t.Errorf("%q: not normalised (|v|²=%.5f)", f.Text, n)
		}
	}
}
