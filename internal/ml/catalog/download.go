package catalog

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"time"
)

// DefaultHubEndpoint is the Hugging Face Hub. HF_ENDPOINT overrides it, which
// is the convention the official clients use for mirrors.
const DefaultHubEndpoint = "https://huggingface.co"

// Hub downloads pinned files from a Hugging Face-compatible Hub.
type Hub struct {
	Endpoint string
	Client   *http.Client
	// Token is an optional access token (HF_TOKEN).
	Token string
}

// NewHub returns a Hub for DefaultHubEndpoint, or HF_ENDPOINT / HF_TOKEN
// from the environment.
func NewHub() *Hub {
	ep := strings.TrimRight(strings.TrimSpace(os.Getenv("HF_ENDPOINT")), "/")
	if ep == "" {
		ep = DefaultHubEndpoint
	}
	return &Hub{Endpoint: ep, Client: http.DefaultClient, Token: strings.TrimSpace(os.Getenv("HF_TOKEN"))}
}

// FileURL is the download URL of a file at the model's pinned revision.
func (h *Hub) FileURL(m Model, f File) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", h.Endpoint, m.Repo, m.Revision, f.Path)
}

// Installed describes a model present on disk, as recorded in the manifest
// written next to its files once every file verified.
type Installed struct {
	ID        string    `json:"id"`
	Repo      string    `json:"repo"`
	Revision  string    `json:"revision"`
	SizeBytes int64     `json:"size_bytes"`
	Installed time.Time `json:"installed"`
	// Dir is the absolute model directory; filled in on load, not stored.
	Dir string `json:"-"`
}

// Path returns the local path of the model's file with the given role.
func (i Installed) Path(role string) string {
	return filepath.Join(i.Dir, File{Role: role}.LocalName())
}

// ModelDir is where a model's files live under the ml data dir.
func ModelDir(dataDir, id string) string {
	return filepath.Join(dataDir, "models", id)
}

func manifestPath(dir string) string { return filepath.Join(dir, "manifest.json") }

// LoadInstalled reads a model's manifest, or returns ok=false when the model
// is absent, incomplete, or was installed from a different revision than
// the catalog now pins.
func LoadInstalled(dataDir string, m Model) (Installed, bool) {
	dir := ModelDir(dataDir, m.ID)
	data, err := os.ReadFile(manifestPath(dir))
	if err != nil {
		return Installed{}, false
	}
	var inst Installed
	if err := json.Unmarshal(data, &inst); err != nil || inst.Revision != m.Revision {
		return Installed{}, false
	}
	for _, f := range m.Files {
		st, err := os.Stat(filepath.Join(dir, f.LocalName()))
		if err != nil || st.Size() != f.Size {
			return Installed{}, false
		}
	}
	if abs, err := filepath.Abs(dir); err == nil {
		dir = abs
	}
	inst.Dir = dir
	return inst, true
}

// ListInstalled returns every catalog model installed under dataDir.
func ListInstalled(dataDir string) []Installed {
	var out []Installed
	for _, m := range All() {
		if inst, ok := LoadInstalled(dataDir, m); ok {
			out = append(out, inst)
		}
	}
	return out
}

// Remove deletes a model's files.
func Remove(dataDir, id string) error {
	if _, ok := Lookup(id); !ok {
		return fmt.Errorf("unknown model %q", id)
	}
	return os.RemoveAll(ModelDir(dataDir, id))
}

// Progress is the live state of a download, safe to read concurrently. One
// Progress can follow several models downloaded one after another.
type Progress struct {
	total atomic.Int64
	done  atomic.Int64
	file  atomic.Pointer[string]
}

// Total is the number of bytes the download needs.
func (p *Progress) Total() int64 { return p.total.Load() }

// Done is the number of bytes present so far.
func (p *Progress) Done() int64 { return p.done.Load() }

// Current is the file currently downloading.
func (p *Progress) Current() string {
	if s := p.file.Load(); s != nil {
		return *s
	}
	return ""
}

// Download fetches a model's files into dataDir, resuming partial files,
// verifying each against its pinned SHA-256, and writing the manifest last
// so a half-finished download never reads as installed. An installed model
// is returned without network access.
func (h *Hub) Download(ctx context.Context, dataDir string, m Model, prog *Progress) (Installed, error) {
	if inst, ok := LoadInstalled(dataDir, m); ok {
		if prog != nil {
			prog.total.Add(m.SizeBytes())
			prog.done.Add(m.SizeBytes())
		}
		return inst, nil
	}
	dir := ModelDir(dataDir, m.ID)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Installed{}, err
	}
	if prog != nil {
		prog.total.Add(m.SizeBytes())
	}
	for _, f := range m.Files {
		if prog != nil {
			name := f.Path
			prog.file.Store(&name)
		}
		dest := filepath.Join(dir, f.LocalName())
		if err := h.fetchFile(ctx, h.FileURL(m, f), dest, f, prog); err != nil {
			return Installed{}, fmt.Errorf("download %s/%s: %w", m.Repo, f.Path, err)
		}
	}
	inst := Installed{ID: m.ID, Repo: m.Repo, Revision: m.Revision, SizeBytes: m.SizeBytes(), Installed: time.Now().UTC()}
	data, _ := json.MarshalIndent(inst, "", "  ")
	tmp := manifestPath(dir) + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return Installed{}, err
	}
	if err := os.Rename(tmp, manifestPath(dir)); err != nil {
		return Installed{}, err
	}
	inst, ok := LoadInstalled(dataDir, m)
	if !ok {
		return Installed{}, fmt.Errorf("model %s did not verify after download", m.ID)
	}
	return inst, nil
}

// fetchFile downloads url to dest with resume (HTTP Range on a .part file)
// and digest verification.
func (h *Hub) fetchFile(ctx context.Context, url, dest string, want File, prog *Progress) error {
	// Already complete and verified by a previous run.
	if st, err := os.Stat(dest); err == nil && st.Size() == want.Size && fileSHA256(dest) == want.SHA256 {
		if prog != nil {
			prog.done.Add(st.Size())
		}
		return nil
	}
	_ = os.Remove(dest)
	part := dest + ".part"
	var offset int64
	if st, err := os.Stat(part); err == nil {
		offset = st.Size()
		if offset >= want.Size {
			offset = 0
			_ = os.Remove(part)
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if h.Token != "" {
		req.Header.Set("Authorization", "Bearer "+h.Token)
	}
	req.Header.Set("User-Agent", "onnx-web-filter")
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	client := h.Client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	flags := os.O_CREATE | os.O_WRONLY
	switch resp.StatusCode {
	case http.StatusPartialContent:
		flags |= os.O_APPEND
	case http.StatusOK:
		offset = 0
		flags |= os.O_TRUNC
	default:
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	if prog != nil {
		prog.done.Add(offset)
	}
	// Never write more than the pinned size, whatever the server sends.
	body := io.LimitReader(resp.Body, want.Size-offset+1)
	buf := make([]byte, 1<<20)
	for {
		n, rerr := body.Read(buf)
		if n > 0 {
			if _, werr := f.Write(buf[:n]); werr != nil {
				f.Close()
				return werr
			}
			if prog != nil {
				prog.done.Add(int64(n))
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			f.Close()
			return rerr
		}
	}
	if err := f.Close(); err != nil {
		return err
	}
	if st, err := os.Stat(part); err != nil || st.Size() != want.Size {
		_ = os.Remove(part)
		return fmt.Errorf("size mismatch: want %d bytes", want.Size)
	}
	if got := fileSHA256(part); got != want.SHA256 {
		_ = os.Remove(part)
		return fmt.Errorf("sha256 mismatch: got %s, want %s", got, want.SHA256)
	}
	return os.Rename(part, dest)
}

func fileSHA256(path string) string {
	f, err := os.Open(path)
	if err != nil {
		return ""
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return ""
	}
	return hex.EncodeToString(h.Sum(nil))
}

// ErrNotInstalled is returned when a model is configured but absent.
var ErrNotInstalled = errors.New("model not installed")
