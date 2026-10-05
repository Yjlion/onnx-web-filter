package ortrt

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
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
	"runtime"
	"strings"
	"time"
)

// Installed records an unpacked runtime.
type Installed struct {
	Version   string    `json:"version"`
	Accel     Accel     `json:"accel"`
	OS        string    `json:"os"`
	Arch      string    `json:"arch"`
	Asset     string    `json:"asset"`
	Library   string    `json:"library"` // absolute path to the shared library
	Installed time.Time `json:"installed"`
}

func manifestPath(dir string) string { return filepath.Join(dir, "manifest.json") }

// LoadInstalled returns the runtime at dataDir for version/accel if its
// library is present.
func LoadInstalled(dataDir, version string, accel Accel) (Installed, bool) {
	dir := InstallDir(dataDir, version, accel)
	data, err := os.ReadFile(manifestPath(dir))
	if err != nil {
		return Installed{}, false
	}
	var m Installed
	if err := json.Unmarshal(data, &m); err != nil {
		return Installed{}, false
	}
	// The library's place inside the install dir is what counts: the
	// manifest's absolute path is stale once the data dir is moved, and
	// points at the original if it was copied.
	lib, ok := FindLibrary(dir)
	if !ok {
		return Installed{}, false
	}
	m.Library = lib
	if abs, err := filepath.Abs(m.Library); err == nil {
		m.Library = abs
	}
	return m, true
}

// Download fetches and unpacks the release archive for this machine. It is
// idempotent: an already-installed runtime is returned without a fetch.
func Download(ctx context.Context, dataDir, version string, accel Accel, report func(string)) (Installed, error) {
	if m, ok := LoadInstalled(dataDir, version, accel); ok {
		return m, nil
	}
	asset, ok := AssetFor(version, runtime.GOOS, runtime.GOARCH, accel)
	if !ok {
		return Installed{}, fmt.Errorf("no prebuilt ONNX Runtime %s for %s/%s (%s)", version, runtime.GOOS, runtime.GOARCH, accel)
	}
	dir := InstallDir(dataDir, version, accel)
	_ = os.RemoveAll(dir)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return Installed{}, err
	}
	if report != nil {
		report("downloading " + asset.Name)
	}
	tmp, err := fetchTemp(ctx, asset.URL, dir, pinnedSHA256[asset.Name])
	if err != nil {
		return Installed{}, err
	}
	if report != nil {
		report("unpacking " + asset.Name)
	}
	err = unpack(tmp, dir)
	_ = os.Remove(tmp)
	if err != nil {
		return Installed{}, fmt.Errorf("unpack %s: %w", asset.Name, err)
	}
	lib, ok := FindLibrary(dir)
	if !ok {
		return Installed{}, fmt.Errorf("%s not found in %s after unpacking", LibraryName(), asset.Name)
	}
	if abs, err := filepath.Abs(lib); err == nil {
		lib = abs
	}
	m := Installed{Version: version, Accel: accel, OS: runtime.GOOS, Arch: runtime.GOARCH, Asset: asset.Name, Library: lib, Installed: time.Now().UTC()}
	data, _ := json.MarshalIndent(m, "", "  ")
	if err := os.WriteFile(manifestPath(dir), data, 0o644); err != nil {
		return Installed{}, err
	}
	return m, nil
}

// fetchTemp downloads url into a temp file in dir. A non-empty wantSHA256
// is checked against the downloaded bytes.
func fetchTemp(ctx context.Context, url, dir, wantSHA256 string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "onnx-web-filter")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download %s: HTTP %d", url, resp.StatusCode)
	}
	f, err := os.CreateTemp(dir, "dl-*"+archiveExt(url))
	if err != nil {
		return "", err
	}
	h := sha256.New()
	if _, err := io.Copy(io.MultiWriter(f, h), resp.Body); err != nil {
		f.Close()
		_ = os.Remove(f.Name())
		return "", err
	}
	if err := f.Close(); err != nil {
		return "", err
	}
	if got := hex.EncodeToString(h.Sum(nil)); wantSHA256 != "" && got != wantSHA256 {
		_ = os.Remove(f.Name())
		return "", fmt.Errorf("download %s: sha256 %s, want %s", url, got, wantSHA256)
	}
	return f.Name(), nil
}

func archiveExt(name string) string {
	switch {
	case strings.HasSuffix(name, ".zip"):
		return ".zip"
	case strings.HasSuffix(name, ".tar.gz"), strings.HasSuffix(name, ".tgz"):
		return ".tar.gz"
	}
	return filepath.Ext(name)
}

// unpack extracts a .zip or .tar.gz/.tgz into dir, refusing paths that escape it.
func unpack(archive, dir string) error {
	switch archiveExt(archive) {
	case ".zip":
		return unzip(archive, dir)
	case ".tar.gz":
		return untargz(archive, dir)
	}
	return errors.New("unsupported archive type")
}

// safeJoin resolves an archive entry name under dir. Rooted names ("/x",
// "\x", "C:x") are refused on every OS, even where joining would keep them
// inside dir: an archive that contains them is not a release archive.
func safeJoin(dir, name string) (string, error) {
	clean := filepath.Clean(filepath.FromSlash(name))
	if clean == "." || strings.HasPrefix(clean, "..") || filepath.IsAbs(clean) ||
		strings.HasPrefix(name, "/") || strings.HasPrefix(name, `\`) || filepath.VolumeName(clean) != "" {
		return "", fmt.Errorf("unsafe path in archive: %q", name)
	}
	p := filepath.Join(dir, clean)
	if !strings.HasPrefix(p, filepath.Clean(dir)+string(os.PathSeparator)) {
		return "", fmt.Errorf("unsafe path in archive: %q", name)
	}
	return p, nil
}

func unzip(archive, dir string) error {
	zr, err := zip.OpenReader(archive)
	if err != nil {
		return err
	}
	defer zr.Close()
	for _, f := range zr.File {
		p, err := safeJoin(dir, f.Name)
		if err != nil {
			return err
		}
		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
			continue
		}
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return err
		}
		rc, err := f.Open()
		if err != nil {
			return err
		}
		err = writeFile(p, rc, f.Mode())
		rc.Close()
		if err != nil {
			return err
		}
	}
	return nil
}

func untargz(archive, dir string) error {
	f, err := os.Open(archive)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		p, err := safeJoin(dir, h.Name)
		if err != nil {
			return err
		}
		switch h.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(p, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return err
			}
			if err := writeFile(p, tr, os.FileMode(h.Mode)); err != nil {
				return err
			}
		case tar.TypeSymlink:
			// Shared libraries ship as versioned symlinks; recreate them
			// relative to the entry, never pointing outside dir.
			if _, err := safeJoin(filepath.Dir(p), h.Linkname); err != nil {
				return err
			}
			_ = os.Remove(p)
			if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
				return err
			}
			if err := os.Symlink(h.Linkname, p); err != nil {
				return err
			}
		}
	}
}

func writeFile(p string, r io.Reader, mode os.FileMode) error {
	if mode&0o111 != 0 {
		mode = 0o755
	} else {
		mode = 0o644
	}
	out, err := os.OpenFile(p, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, mode)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, r); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
