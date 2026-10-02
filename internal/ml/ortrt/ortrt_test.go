package ortrt

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestAssetForCoversSupportedPlatforms(t *testing.T) {
	cases := []struct {
		goos, arch string
		accel      Accel
		want       string
	}{
		{"linux", "amd64", AccelCPU, "onnxruntime-linux-x64-1.2.3.tgz"},
		{"linux", "amd64", AccelCUDA, "onnxruntime-linux-x64-gpu_cuda12-1.2.3.tgz"},
		{"linux", "amd64", AccelCUDA13, "onnxruntime-linux-x64-gpu_cuda13-1.2.3.tgz"},
		{"linux", "arm64", AccelCPU, "onnxruntime-linux-aarch64-1.2.3.tgz"},
		{"darwin", "arm64", AccelCPU, "onnxruntime-osx-arm64-1.2.3.tgz"},
		{"windows", "amd64", AccelCPU, "onnxruntime-win-x64-1.2.3.zip"},
		{"windows", "amd64", AccelCUDA, "onnxruntime-win-x64-gpu_cuda12-1.2.3.zip"},
		{"windows", "arm64", AccelCPU, "onnxruntime-win-arm64-1.2.3.zip"},
	}
	for _, c := range cases {
		a, ok := AssetFor("1.2.3", c.goos, c.arch, c.accel)
		if !ok || a.Name != c.want {
			t.Errorf("AssetFor(%s/%s/%s) = %q,%v want %q", c.goos, c.arch, c.accel, a.Name, ok, c.want)
		}
		if !strings.HasSuffix(a.URL, "/v1.2.3/"+c.want) {
			t.Errorf("URL %q should end with /v1.2.3/%s", a.URL, c.want)
		}
	}
	for _, bad := range []struct {
		goos, arch string
		accel      Accel
	}{
		{"freebsd", "amd64", AccelCPU},
		{"darwin", "amd64", AccelCPU}, // no macOS x64 build since 1.30
		{"linux", "arm64", AccelCUDA},
		{"darwin", "arm64", AccelCUDA},
		{"linux", "amd64", Accel("vulkan")},
	} {
		if _, ok := AssetFor("1.2.3", bad.goos, bad.arch, bad.accel); ok {
			t.Errorf("AssetFor(%s/%s/%s) must report ok=false", bad.goos, bad.arch, bad.accel)
		}
	}
}

func TestEveryPinnedAssetHasChecksum(t *testing.T) {
	for _, p := range []struct{ goos, arch string }{
		{"linux", "amd64"}, {"linux", "arm64"}, {"darwin", "arm64"}, {"windows", "amd64"}, {"windows", "arm64"},
	} {
		for _, accel := range []Accel{AccelCPU, AccelCUDA, AccelCUDA13} {
			a, ok := AssetFor(PinnedVersion, p.goos, p.arch, accel)
			if !ok {
				continue
			}
			if len(pinnedSHA256[a.Name]) != 64 {
				t.Errorf("no pinned sha256 for %s", a.Name)
			}
		}
	}
}

func TestResolveAccel(t *testing.T) {
	for in, want := range map[string]Accel{"cpu": AccelCPU, "CUDA": AccelCUDA, "cuda12": AccelCUDA, "cuda13": AccelCUDA13} {
		if got := ResolveAccel(in); got != want {
			t.Errorf("ResolveAccel(%q) = %s, want %s", in, got, want)
		}
	}
}

func TestSafeJoinRejectsEscapes(t *testing.T) {
	dir := t.TempDir()
	for _, bad := range []string{"../x", "/etc/passwd", "a/../../b"} {
		if _, err := safeJoin(dir, bad); err == nil {
			t.Errorf("safeJoin(%q) should fail", bad)
		}
	}
	if _, err := safeJoin(dir, "onnxruntime-linux-x64-1.30.0/lib/libonnxruntime.so"); err != nil {
		t.Errorf("safeJoin(normal) = %v", err)
	}
}

type entry struct {
	name, body, link string
}

// makeTgz builds an archive shaped like the real release: a versioned
// library plus the unversioned name as a symlink chain.
func makeTgz(t *testing.T, entries []entry) []byte {
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	for _, e := range entries {
		h := &tar.Header{Name: e.name, Mode: 0o755, Size: int64(len(e.body)), Typeflag: tar.TypeReg}
		if e.link != "" {
			h = &tar.Header{Name: e.name, Linkname: e.link, Mode: 0o777, Typeflag: tar.TypeSymlink}
		}
		if err := tw.WriteHeader(h); err != nil {
			t.Fatal(err)
		}
		if e.link == "" {
			if _, err := tw.Write([]byte(e.body)); err != nil {
				t.Fatal(err)
			}
		}
	}
	_ = tw.Close()
	_ = gz.Close()
	return buf.Bytes()
}

func makeZip(t *testing.T, files map[string]string) []byte {
	var buf bytes.Buffer
	zw := zip.NewWriter(&buf)
	for name, body := range files {
		w, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = w.Write([]byte(body))
	}
	_ = zw.Close()
	return buf.Bytes()
}

func fakeRelease(t *testing.T, asset Asset) []byte {
	root := strings.TrimSuffix(strings.TrimSuffix(asset.Name, ".tgz"), ".zip")
	if strings.HasSuffix(asset.Name, ".zip") {
		return makeZip(t, map[string]string{root + "/lib/" + LibraryName(): "x", root + "/LICENSE": "MIT"})
	}
	lib := LibraryName()
	return makeTgz(t, []entry{
		{name: root + "/lib/" + lib + ".1.2.3", body: "x"},
		{name: root + "/lib/" + lib + ".1", link: lib + ".1.2.3"},
		{name: root + "/lib/" + lib, link: lib + ".1"},
		{name: root + "/LICENSE", body: "MIT"},
	})
}

func TestDownloadUnpacksAndFindsLibrary(t *testing.T) {
	accel := AccelCPU
	asset, ok := AssetFor("1.2.3", runtime.GOOS, runtime.GOARCH, accel)
	if !ok {
		t.Skip("no asset for this platform")
	}
	payload := fakeRelease(t, asset)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/v1.2.3/"+asset.Name) {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(payload)
	}))
	defer ts.Close()
	t.Setenv("ORT_RELEASE_BASE", ts.URL)

	dataDir := t.TempDir()
	var steps []string
	inst, err := Download(context.Background(), dataDir, "1.2.3", accel, func(s string) { steps = append(steps, s) })
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	if data, err := os.ReadFile(inst.Library); err != nil || string(data) != "x" {
		t.Fatalf("library not readable through its symlinks: %q, %v", data, err)
	}
	if !strings.HasPrefix(inst.Library, InstallDir(dataDir, "1.2.3", accel)) {
		t.Errorf("library %q not under install dir", inst.Library)
	}
	if len(steps) != 2 {
		t.Errorf("expected download+unpack progress steps, got %v", steps)
	}
	if got, ok := LoadInstalled(dataDir, "1.2.3", accel); !ok || got.Library != inst.Library {
		t.Fatalf("LoadInstalled = %+v, %v", got, ok)
	}
	// Second call is served from the manifest, no network.
	ts.Close()
	if _, err := Download(context.Background(), dataDir, "1.2.3", accel, nil); err != nil {
		t.Fatalf("second Download should be a no-op: %v", err)
	}
	if _, ok := FindLibrary(filepath.Dir(inst.Library)); !ok {
		t.Error("FindLibrary failed on the lib dir")
	}
}

func TestFetchTempVerifiesChecksum(t *testing.T) {
	body := []byte("archive bytes")
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(body) }))
	defer ts.Close()
	dir := t.TempDir()
	sum := sha256.Sum256(body)
	p, err := fetchTemp(context.Background(), ts.URL+"/a.tgz", dir, hex.EncodeToString(sum[:]))
	if err != nil {
		t.Fatalf("matching checksum rejected: %v", err)
	}
	_ = os.Remove(p)
	if _, err := fetchTemp(context.Background(), ts.URL+"/a.tgz", dir, strings.Repeat("0", 64)); err == nil {
		t.Fatal("mismatching checksum accepted")
	}
	if left, _ := os.ReadDir(dir); len(left) != 0 {
		t.Errorf("rejected download left files behind: %v", left)
	}
}
