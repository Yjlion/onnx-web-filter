// Package classify runs the catalog's ONNX models: adult-image and
// adult-text classifiers and the sentence embedder behind site categories.
// ortenv.Init must have loaded the runtime first.
//
// Every type here is safe for concurrent use: an ONNX Runtime session runs
// concurrent Run calls, and the Go side keeps no per-call state.
package classify

import (
	"context"
	"fmt"
	"math"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"
)

// Options configure the sessions a model opens.
type Options struct {
	// Threads is the intra-op thread count per session; 0 lets ONNX
	// Runtime use every core. With several verdict workers running models
	// at once, cores/workers avoids oversubscription.
	Threads int
	// GPU appends the CUDA execution provider; if that fails (no CUDA or
	// cuDNN libraries, no device) the session is built for the CPU and
	// Provider says why.
	GPU      bool
	DeviceID int
}

// Provider says where a session runs: "cpu", "cuda", or
// "cpu (cuda failed: ...)".
type Provider string

func openSession(path string, o Options) (*ort.Session, Provider, error) {
	if o.GPU {
		s, err := newSession(path, o, true)
		if err == nil {
			return s, "cuda", nil
		}
		s, cpuErr := newSession(path, o, false)
		if cpuErr != nil {
			return nil, "", cpuErr
		}
		return s, Provider("cpu (cuda failed: " + err.Error() + ")"), nil
	}
	s, err := newSession(path, o, false)
	return s, "cpu", err
}

func newSession(path string, o Options, cuda bool) (*ort.Session, error) {
	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}
	defer opts.Close()
	if o.Threads > 0 {
		if err := opts.SetIntraOpNumThreads(o.Threads); err != nil {
			return nil, err
		}
	}
	if err := opts.SetGraphOptimizationLevel(ort.GraphOptimizationLevelAll); err != nil {
		return nil, err
	}
	if cuda {
		if err := opts.AppendExecutionProvider("CUDA", map[string]string{"device_id": fmt.Sprint(o.DeviceID)}); err != nil {
			return nil, err
		}
	}
	s, err := ort.NewSession(path, opts)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	return s, nil
}

// checkIO verifies the session has the named inputs and output, so a
// catalog mistake fails at load time instead of on the first verdict.
func checkIO(s *ort.Session, inputs []string, output string) error {
	have := map[string]bool{}
	for _, in := range s.Inputs() {
		have[in.Name] = true
	}
	for _, name := range inputs {
		if !have[name] {
			return fmt.Errorf("model has no input %q (has %v)", name, s.Inputs())
		}
	}
	for _, out := range s.Outputs() {
		if out.Name == output {
			return nil
		}
	}
	return fmt.Errorf("model has no output %q (has %v)", output, s.Outputs())
}

// runFloat runs s and returns a copy of one float32 output and its shape.
// It closes every tensor it is handed or receives.
func runFloat(ctx context.Context, s *ort.Session, inputs map[string]*ort.Tensor, output string) ([]float32, []int64, error) {
	defer func() {
		for _, t := range inputs {
			_ = t.Close()
		}
	}()
	outs, err := s.Run(ctx, inputs, []string{output})
	if err != nil {
		return nil, nil, err
	}
	defer func() {
		for _, t := range outs {
			_ = t.Close()
		}
	}()
	t, ok := outs[output]
	if !ok {
		return nil, nil, fmt.Errorf("output %q missing from run result", output)
	}
	data, err := ort.TensorData[float32](t)
	if err != nil {
		return nil, nil, err
	}
	return append([]float32(nil), data...), t.Shape(), nil
}

// Scores is a classifier's answer for one input.
type Scores struct {
	Labels []string  `json:"labels"`
	Probs  []float64 `json:"probs"`
	// Unsafe is the adult score in [0,1]: the catalog's unsafe weights
	// applied to Probs.
	Unsafe float64 `json:"unsafe"`
}

// Top is the most probable label.
func (s Scores) Top() (string, float64) {
	best := -1
	for i, p := range s.Probs {
		if best < 0 || p > s.Probs[best] {
			best = i
		}
	}
	if best < 0 {
		return "", 0
	}
	return s.Labels[best], s.Probs[best]
}

// scores turns one row of model output into Scores.
func scores(row []float32, softmax bool, labels []string, weights []float64) (Scores, error) {
	if len(row) != len(labels) {
		return Scores{}, fmt.Errorf("model gave %d outputs for %d labels", len(row), len(labels))
	}
	probs := make([]float64, len(row))
	for i, v := range row {
		probs[i] = float64(v)
	}
	if softmax {
		probs = softmaxF(probs)
	}
	var u float64
	for i, p := range probs {
		u += weights[i] * p
	}
	return Scores{Labels: labels, Probs: probs, Unsafe: min(max(u, 0), 1)}, nil
}

func softmaxF(x []float64) []float64 {
	m := math.Inf(-1)
	for _, v := range x {
		m = max(m, v)
	}
	var sum float64
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Exp(v - m)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
