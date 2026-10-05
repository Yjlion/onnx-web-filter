package catalog

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
)

func TestEmbeddedCatalogIsValid(t *testing.T) {
	if len(All()) < 3 {
		t.Fatalf("catalog has %d models", len(All()))
	}
	for _, task := range []Task{TaskImage, TaskText, TaskSite} {
		m := Default(task)
		if got, err := Resolve(task, ""); err != nil || got.ID != m.ID {
			t.Errorf("Resolve(%s, \"\") = %v, %v", task, got.ID, err)
		}
	}
	if _, err := Resolve(TaskText, Default(TaskImage).ID); err == nil {
		t.Error("an image model must not resolve for the text task")
	}
	if _, err := Resolve(TaskImage, "nope"); err == nil {
		t.Error("unknown id must fail")
	}
	img := Default(TaskImage)
	w := img.UnsafeWeights()
	if len(w) != len(img.Labels) {
		t.Fatalf("weights %v for labels %v", w, img.Labels)
	}
}

func TestValidateRejectsUnpinned(t *testing.T) {
	good := Model{
		ID: "x", Task: TaskImage, Revision: strings.Repeat("a", 40),
		Files:  []File{{Role: "model", Path: "m.onnx", Size: 1, SHA256: strings.Repeat("0", 64)}},
		Image:  &ImageSpec{Layout: "NCHW"},
		Labels: []string{"ok", "bad"}, Unsafe: map[string]float64{"bad": 1},
	}
	if err := validateOne(good); err != nil {
		t.Fatalf("good model rejected: %v", err)
	}
	bad := good
	bad.Files = []File{{Role: "model", Path: "m.onnx", Size: 1}}
	if validateOne(bad) == nil {
		t.Error("file without sha256 accepted")
	}
	bad = good
	bad.Revision = "main"
	if validateOne(bad) == nil {
		t.Error("branch revision accepted")
	}
	bad = good
	bad.Unsafe = map[string]float64{"typo": 1}
	if validateOne(bad) == nil {
		t.Error("unsafe weight for an unknown label accepted")
	}
}

// validateOne validates m as the only model of its task.
func validateOne(m Model) error {
	m.Default = true
	ms := []Model{m}
	for _, task := range []Task{TaskImage, TaskText, TaskSite} {
		if task == m.Task {
			continue
		}
		stub := Model{
			ID: "stub-" + string(task), Task: task, Revision: strings.Repeat("b", 40), Default: true,
			Files: []File{
				{Role: "model", Path: "m.onnx", Size: 1, SHA256: strings.Repeat("0", 64)},
				{Role: "tokenizer", Path: "tokenizer.json", Size: 1, SHA256: strings.Repeat("0", 64)},
			},
			Text: &TextSpec{}, Embed: &EmbedSpec{Pooling: "mean"},
			Labels: []string{"a"}, Unsafe: map[string]float64{"a": 1},
		}
		ms = append(ms, stub)
	}
	return validate(ms)
}

type fakeHub struct {
	files    map[string][]byte // repo path -> content
	requests atomic.Int64
	// cut, if set, makes the next response stop after this many bytes.
	cut atomic.Int64
}

func (f *fakeHub) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.requests.Add(1)
	// /<owner>/<repo>/resolve/<rev>/<path>
	parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 5)
	if len(parts) != 5 || parts[2] != "resolve" {
		http.NotFound(w, r)
		return
	}
	body, ok := f.files[parts[4]]
	if !ok {
		http.NotFound(w, r)
		return
	}
	status := http.StatusOK
	if rg := r.Header.Get("Range"); rg != "" {
		off, _ := strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rg, "bytes="), "-"))
		body = body[off:]
		status = http.StatusPartialContent
	}
	if c := f.cut.Swap(0); c > 0 && int(c) < len(body) {
		w.Header().Set("Content-Length", strconv.Itoa(len(body)))
		w.WriteHeader(status)
		_, _ = w.Write(body[:c])
		// Hijack-free way to end early: the client sees an unexpected EOF.
		return
	}
	w.WriteHeader(status)
	_, _ = w.Write(body)
}

func pinned(path, role string, body []byte) File {
	sum := sha256.Sum256(body)
	return File{Role: role, Path: path, Size: int64(len(body)), SHA256: hex.EncodeToString(sum[:])}
}

func testModel(files map[string][]byte) Model {
	return Model{
		ID: "test-model", Task: TaskText, Repo: "acme/test", Revision: strings.Repeat("c", 40),
		Files: []File{
			pinned("onnx/model.onnx", "model", files["onnx/model.onnx"]),
			pinned("tokenizer.json", "tokenizer", files["tokenizer.json"]),
		},
	}
}

func TestDownloadVerifiesAndIsIdempotent(t *testing.T) {
	files := map[string][]byte{"onnx/model.onnx": []byte(strings.Repeat("onnx", 1000)), "tokenizer.json": []byte(`{"model":{}}`)}
	fh := &fakeHub{files: files}
	ts := httptest.NewServer(fh)
	defer ts.Close()
	hub := &Hub{Endpoint: ts.URL}
	m := testModel(files)
	data := t.TempDir()

	prog := &Progress{}
	inst, err := hub.Download(context.Background(), data, m, prog)
	if err != nil {
		t.Fatal(err)
	}
	if prog.Done() != m.SizeBytes() || prog.Total() != m.SizeBytes() {
		t.Errorf("progress %d/%d, want %d", prog.Done(), prog.Total(), m.SizeBytes())
	}
	got, err := os.ReadFile(inst.Path("model"))
	if err != nil || string(got) != string(files["onnx/model.onnx"]) {
		t.Fatalf("model file: %v", err)
	}
	if _, err := os.Stat(inst.Path("tokenizer")); err != nil {
		t.Fatal(err)
	}
	if l := ListInstalled(data); len(l) != 0 {
		t.Errorf("ListInstalled only lists catalog models, got %v", l)
	}

	before := fh.requests.Load()
	if _, err := hub.Download(context.Background(), data, m, nil); err != nil {
		t.Fatal(err)
	}
	if fh.requests.Load() != before {
		t.Error("installed model was downloaded again")
	}

	// A new pinned revision makes the old install stale.
	m2 := m
	m2.Revision = strings.Repeat("d", 40)
	if _, ok := LoadInstalled(data, m2); ok {
		t.Error("install of an older revision reported as current")
	}
}

func TestDownloadRejectsTamperedFile(t *testing.T) {
	files := map[string][]byte{"onnx/model.onnx": []byte("genuine model"), "tokenizer.json": []byte("{}")}
	m := testModel(files)
	files["onnx/model.onnx"] = []byte("evil  model!!") // same size, different bytes
	ts := httptest.NewServer(&fakeHub{files: files})
	defer ts.Close()
	data := t.TempDir()
	_, err := (&Hub{Endpoint: ts.URL}).Download(context.Background(), data, m, nil)
	if err == nil || !strings.Contains(err.Error(), "sha256") {
		t.Fatalf("tampered file: err = %v", err)
	}
	if _, ok := LoadInstalled(data, m); ok {
		t.Fatal("tampered model reads as installed")
	}
	left, _ := filepath.Glob(filepath.Join(ModelDir(data, m.ID), "*"))
	for _, p := range left {
		if strings.HasSuffix(p, "model.onnx") || strings.HasSuffix(p, ".part") {
			t.Errorf("tampered bytes left on disk: %s", p)
		}
	}
}

func TestDownloadResumes(t *testing.T) {
	body := []byte(strings.Repeat("0123456789", 5000))
	files := map[string][]byte{"onnx/model.onnx": body, "tokenizer.json": []byte("{}")}
	fh := &fakeHub{files: files}
	fh.cut.Store(12345)
	ts := httptest.NewServer(fh)
	defer ts.Close()
	hub := &Hub{Endpoint: ts.URL}
	m := testModel(files)
	data := t.TempDir()

	if _, err := hub.Download(context.Background(), data, m, nil); err == nil {
		t.Fatal("cut-short download reported success")
	}
	part := filepath.Join(ModelDir(data, m.ID), "model.onnx.part")
	if st, err := os.Stat(part); err != nil || st.Size() != 12345 {
		t.Fatalf("partial file: %v %v", st, err)
	}
	if _, err := hub.Download(context.Background(), data, m, nil); err != nil {
		t.Fatalf("resume: %v", err)
	}
	if _, ok := LoadInstalled(data, m); !ok {
		t.Fatal("resumed model not installed")
	}
}

func TestFileURLIsPinned(t *testing.T) {
	m := Default(TaskImage)
	f, _ := m.File("model")
	u := (&Hub{Endpoint: "https://hub.example"}).FileURL(m, f)
	want := fmt.Sprintf("https://hub.example/%s/resolve/%s/%s", m.Repo, m.Revision, f.Path)
	if u != want {
		t.Errorf("FileURL = %s, want %s", u, want)
	}
}
