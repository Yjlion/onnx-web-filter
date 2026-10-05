package models

import (
	"encoding/json"
	"strings"
)

// MLConfig configures in-process content classification with ONNX Runtime:
// adult images, adult page text and site categories.
//
// The filter downloads a prebuilt ONNX Runtime release and the catalog's
// models (internal/ml/catalog) on first run, into DataDir, and loads them
// into its own process; there is no model server.
type MLConfig struct {
	// Enabled turns classification on. Off means the classifier addons pass
	// content through (keyword-only for text) and sites are categorised by
	// the domain lists alone; the proxy itself still works.
	Enabled bool `json:"enabled"`

	// DataDir holds the downloaded runtime, the models and the decision
	// cache. Empty means <project root>/data/ml, resolved at bootstrap like
	// the other dirs.
	DataDir string `json:"data_dir"`

	// Accel selects the ONNX Runtime build: "auto" uses the CUDA 12 build
	// when an NVIDIA driver is present and the CPU build otherwise; "cpu",
	// "cuda" (CUDA 12) and "cuda13" force one. A CUDA session that cannot
	// start falls back to the CPU.
	Accel string `json:"accel"`

	// ORTLibPath, when set, loads this ONNX Runtime shared library (file or
	// directory) instead of downloading one. ORT_LIB_PATH does the same.
	ORTLibPath string `json:"ort_lib_path"`

	// ImageModel, TextModel and SiteModel are catalog ids; empty picks the
	// catalog default for the task.
	ImageModel string `json:"image_model"`
	TextModel  string `json:"text_model"`
	SiteModel  string `json:"site_model"`

	// Threads is the intra-op thread count of each model session; 0 picks
	// cores divided by Parallel, so concurrent verdicts do not fight over
	// cores.
	Threads int `json:"threads"`

	// Parallel is how many verdicts run at once; 0 picks 4 on a GPU and 2
	// on a CPU.
	Parallel int `json:"parallel"`

	// MaxImagePx caps the longest side of an image before it is cached and
	// classified. The models look at 224px, so 384 loses nothing.
	MaxImagePx int `json:"max_image_px"`

	// AdultScore is the score at which a verdict counts as certainly adult:
	// it is blocked whatever the policy threshold, and it counts towards
	// learning a whole site as adult.
	AdultScore float64 `json:"adult_score"`

	// Budget bounds how long a request waits for a verdict before the
	// policy's on_timeout action applies. The job keeps running past the
	// budget so its verdict still lands in the decision cache.
	Budget MLBudget `json:"budget"`
}

// MLBudget is the per-kind wait budget, in milliseconds.
type MLBudget struct {
	ImageMs int `json:"image_ms"`
	TextMs  int `json:"text_ms"`
	// CategoryMs is how long a navigation to a not-yet-categorized site
	// waits for the model's category.
	CategoryMs int `json:"category_ms"`
}

// NewMLConfig returns the documented defaults.
func NewMLConfig() MLConfig {
	return MLConfig{
		Enabled:    true,
		Accel:      "auto",
		MaxImagePx: 384,
		AdultScore: 0.9,
		Budget:     MLBudget{ImageMs: 500, TextMs: 500, CategoryMs: 500},
	}
}

type mlConfigAlias MLConfig

func (c *MLConfig) UnmarshalJSON(data []byte) error {
	*c = NewMLConfig()
	if err := json.Unmarshal(data, (*mlConfigAlias)(c)); err != nil {
		return err
	}
	c.Accel = strings.ToLower(strings.TrimSpace(c.Accel))
	switch c.Accel {
	case "auto", "cpu", "cuda", "cuda12", "cuda13":
	default:
		c.Accel = "auto"
	}
	c.DataDir = strings.TrimSpace(c.DataDir)
	c.ORTLibPath = strings.TrimSpace(c.ORTLibPath)
	c.ImageModel = strings.TrimSpace(c.ImageModel)
	c.TextModel = strings.TrimSpace(c.TextModel)
	c.SiteModel = strings.TrimSpace(c.SiteModel)
	if c.Threads < 0 {
		c.Threads = 0
	}
	if c.Parallel < 0 {
		c.Parallel = 0
	}
	if c.MaxImagePx < 64 {
		c.MaxImagePx = 384
	}
	if c.AdultScore <= 0 || c.AdultScore > 1 {
		c.AdultScore = 0.9
	}
	d := NewMLConfig().Budget
	if c.Budget.ImageMs <= 0 {
		c.Budget.ImageMs = d.ImageMs
	}
	if c.Budget.TextMs <= 0 {
		c.Budget.TextMs = d.TextMs
	}
	if c.Budget.CategoryMs <= 0 {
		c.Budget.CategoryMs = d.CategoryMs
	}
	return nil
}
