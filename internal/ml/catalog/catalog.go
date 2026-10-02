// Package catalog is the list of ONNX models onnx-web-filter knows how to
// download and run. Every file is pinned to a repository commit, a size and
// a SHA-256, so a download is reproducible and a changed upstream file is
// rejected instead of silently loaded.
package catalog

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"sort"
)

// Task is what a model is used for.
type Task string

const (
	TaskImage Task = "image" // adult image score
	TaskText  Task = "text"  // adult page-text score
	TaskSite  Task = "site"  // sentence embedding for site categories
)

// File is one pinned file of a model repository.
type File struct {
	// Role is "model" (the .onnx graph) or "tokenizer" (tokenizer.json).
	Role   string `json:"role"`
	Path   string `json:"path"` // path inside the repository
	Size   int64  `json:"size"`
	SHA256 string `json:"sha256"`
}

// LocalName is the file's name inside the model directory.
func (f File) LocalName() string {
	switch f.Role {
	case "model":
		return "model.onnx"
	case "tokenizer":
		return "tokenizer.json"
	}
	return f.Role
}

// ImageSpec says how to turn a picture into the model's input tensor.
type ImageSpec struct {
	Size int `json:"size"` // square input side
	// Layout is "NCHW" or "NHWC".
	Layout string `json:"layout"`
	// Pixels are scaled to [0,1], then normalised as (x-mean)/std.
	Mean    [3]float32 `json:"mean"`
	Std     [3]float32 `json:"std"`
	Input   string     `json:"input"`
	Output  string     `json:"output"`
	Softmax bool       `json:"softmax"` // output is logits
}

// TextSpec describes a sequence classifier.
type TextSpec struct {
	MaxLen  int      `json:"max_len"`
	Inputs  []string `json:"inputs"` // subset of input_ids, attention_mask, token_type_ids
	Output  string   `json:"output"`
	Softmax bool     `json:"softmax"`
}

// EmbedSpec describes a sentence-embedding model.
type EmbedSpec struct {
	MaxLen  int      `json:"max_len"`
	Inputs  []string `json:"inputs"`
	Output  string   `json:"output"` // [batch, seq, dim] hidden states
	Pooling string   `json:"pooling"`
	// Normalize L2-normalises the pooled vector.
	Normalize bool `json:"normalize"`
}

// Model is one catalog entry.
type Model struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Task        Task   `json:"task"`
	Description string `json:"description"`
	License     string `json:"license"`
	Repo        string `json:"repo"`
	Revision    string `json:"revision"`
	Files       []File `json:"files"`

	Image *ImageSpec `json:"image,omitempty"`
	Text  *TextSpec  `json:"text,omitempty"`
	Embed *EmbedSpec `json:"embed,omitempty"`

	// Labels name the classifier outputs in order.
	Labels []string `json:"labels,omitempty"`
	// Unsafe weights labels into the adult score: score = Σ w·p(label),
	// capped at 1.
	Unsafe map[string]float64 `json:"unsafe,omitempty"`

	// Default marks the model used for its task when none is configured.
	Default bool `json:"default,omitempty"`
}

// SizeBytes is the total download size.
func (m Model) SizeBytes() int64 {
	var n int64
	for _, f := range m.Files {
		n += f.Size
	}
	return n
}

// File returns the file with the given role.
func (m Model) File(role string) (File, bool) {
	for _, f := range m.Files {
		if f.Role == role {
			return f, true
		}
	}
	return File{}, false
}

// UnsafeWeights returns the Unsafe weights in label order.
func (m Model) UnsafeWeights() []float64 {
	w := make([]float64, len(m.Labels))
	for i, l := range m.Labels {
		w[i] = m.Unsafe[l]
	}
	return w
}

//go:embed models.json
var modelsJSON []byte

var models = mustParse(modelsJSON)

func mustParse(data []byte) []Model {
	var ms []Model
	if err := json.Unmarshal(data, &ms); err != nil {
		panic("catalog: models.json: " + err.Error())
	}
	if err := validate(ms); err != nil {
		panic("catalog: models.json: " + err.Error())
	}
	return ms
}

func validate(ms []Model) error {
	seen := map[string]bool{}
	defaults := map[Task]int{}
	for _, m := range ms {
		if m.ID == "" || seen[m.ID] {
			return fmt.Errorf("missing or duplicate id %q", m.ID)
		}
		seen[m.ID] = true
		if len(m.Revision) != 40 {
			return fmt.Errorf("%s: revision must be a full commit hash", m.ID)
		}
		if _, ok := m.File("model"); !ok {
			return fmt.Errorf("%s: no model file", m.ID)
		}
		for _, f := range m.Files {
			if len(f.SHA256) != 64 || f.Size <= 0 {
				return fmt.Errorf("%s: %s is not pinned (size and sha256)", m.ID, f.Path)
			}
		}
		switch m.Task {
		case TaskImage:
			if m.Image == nil || (m.Image.Layout != "NCHW" && m.Image.Layout != "NHWC") {
				return fmt.Errorf("%s: bad image spec", m.ID)
			}
		case TaskText:
			if m.Text == nil {
				return fmt.Errorf("%s: no text spec", m.ID)
			}
		case TaskSite:
			if m.Embed == nil || m.Embed.Pooling != "mean" {
				return fmt.Errorf("%s: bad embed spec (only mean pooling)", m.ID)
			}
		default:
			return fmt.Errorf("%s: unknown task %q", m.ID, m.Task)
		}
		if m.Task == TaskText || m.Task == TaskSite {
			if _, ok := m.File("tokenizer"); !ok {
				return fmt.Errorf("%s: no tokenizer file", m.ID)
			}
		}
		if m.Task != TaskSite {
			if len(m.Labels) == 0 || len(m.Unsafe) == 0 {
				return fmt.Errorf("%s: classifier needs labels and unsafe weights", m.ID)
			}
			for l := range m.Unsafe {
				if !contains(m.Labels, l) {
					return fmt.Errorf("%s: unsafe label %q not in labels", m.ID, l)
				}
			}
		}
		if m.Default {
			defaults[m.Task]++
		}
	}
	for _, t := range []Task{TaskImage, TaskText, TaskSite} {
		if defaults[t] != 1 {
			return fmt.Errorf("task %s needs exactly one default model, has %d", t, defaults[t])
		}
	}
	return nil
}

func contains(xs []string, x string) bool {
	for _, v := range xs {
		if v == x {
			return true
		}
	}
	return false
}

// All returns the catalog, grouped by task.
func All() []Model {
	out := append([]Model(nil), models...)
	order := map[Task]int{TaskImage: 0, TaskText: 1, TaskSite: 2}
	sort.SliceStable(out, func(i, j int) bool { return order[out[i].Task] < order[out[j].Task] })
	return out
}

// Lookup finds a model by id.
func Lookup(id string) (Model, bool) {
	for _, m := range models {
		if m.ID == id {
			return m, true
		}
	}
	return Model{}, false
}

// Default returns the default model for a task.
func Default(t Task) Model {
	for _, m := range models {
		if m.Task == t && m.Default {
			return m
		}
	}
	panic("catalog: no default for " + string(t)) // validate guarantees one
}

// Resolve returns the configured model for a task: id when set (it must
// exist and be for that task), the task default otherwise.
func Resolve(t Task, id string) (Model, error) {
	if id == "" {
		return Default(t), nil
	}
	m, ok := Lookup(id)
	if !ok {
		return Model{}, fmt.Errorf("unknown model %q", id)
	}
	if m.Task != t {
		return Model{}, fmt.Errorf("model %q is a %s model, not %s", id, m.Task, t)
	}
	return m, nil
}
