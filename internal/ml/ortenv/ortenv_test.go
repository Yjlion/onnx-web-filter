package ortenv

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
)

// install fakes a downloaded runtime of the given flavour under dataDir.
func install(t *testing.T, dataDir string, accel ortrt.Accel) string {
	t.Helper()
	dir := ortrt.InstallDir(dataDir, ortrt.PinnedVersion, accel)
	lib := filepath.Join(dir, "onnxruntime-x", "lib", ortrt.LibraryName())
	if err := os.MkdirAll(filepath.Dir(lib), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(lib, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	m, _ := json.Marshal(ortrt.Installed{Version: ortrt.PinnedVersion, Accel: accel, Library: lib})
	if err := os.WriteFile(filepath.Join(dir, "manifest.json"), m, 0o644); err != nil {
		t.Fatal(err)
	}
	return lib
}

func TestLocateOrder(t *testing.T) {
	t.Setenv("ORT_LIB_PATH", "")
	data := t.TempDir()

	if _, err := Locate("", data, ortrt.AccelCPU); !errors.Is(err, ErrNotInstalled) {
		t.Fatalf("empty data dir: err = %v, want ErrNotInstalled", err)
	}

	cpu := install(t, data, ortrt.AccelCPU)
	lib, err := Locate("", data, ortrt.AccelCUDA)
	if err != nil || lib.Path != cpu || lib.Accel != ortrt.AccelCPU || lib.Source != "download" {
		t.Fatalf("missing CUDA build should fall back to CPU: %+v, %v", lib, err)
	}

	cuda := install(t, data, ortrt.AccelCUDA)
	if lib, _ := Locate("", data, ortrt.AccelCUDA); lib.Path != cuda || lib.Accel != ortrt.AccelCUDA {
		t.Fatalf("installed CUDA build not preferred: %+v", lib)
	}

	// ORT_LIB_PATH beats a download; a directory is accepted.
	t.Setenv("ORT_LIB_PATH", filepath.Dir(cpu))
	if lib, err := Locate("", data, ortrt.AccelCUDA); err != nil || lib.Path != cpu || lib.Source != "env" {
		t.Fatalf("ORT_LIB_PATH dir: %+v, %v", lib, err)
	}

	// An explicit setting beats everything; the release root is accepted.
	root := filepath.Dir(filepath.Dir(cuda))
	if lib, err := Locate(root, data, ortrt.AccelCPU); err != nil || lib.Path != cuda || lib.Source != "config" {
		t.Fatalf("explicit release root: %+v, %v", lib, err)
	}
	if _, err := Locate(filepath.Join(data, "nope"), data, ortrt.AccelCPU); err == nil {
		t.Fatal("missing explicit path must fail, not fall through")
	}
}

// TestInitRealRuntime loads the library named by ORT_LIB_PATH, e.g.
//
//	ORT_LIB_PATH=data/ml/runtime/1.30.0-linux-amd64-cpu go test ./internal/ml/ortenv
func TestInitRealRuntime(t *testing.T) {
	if os.Getenv("ORT_LIB_PATH") == "" {
		t.Skip("ORT_LIB_PATH not set")
	}
	lib, err := Locate("", "", ortrt.AccelCPU)
	if err != nil {
		t.Fatal(err)
	}
	info, err := Init(lib)
	if err != nil {
		t.Fatal(err)
	}
	if info.Version == "" || info.APIVersion < 17 {
		t.Errorf("implausible info: %+v", info)
	}
	if !info.HasProvider("CPUExecutionProvider") {
		t.Errorf("CPU provider missing: %v", info.Providers)
	}
	// Idempotent.
	if again, err := Init(lib); err != nil || again.Version != info.Version {
		t.Errorf("second Init = %+v, %v", again, err)
	}
	t.Logf("ONNX Runtime %s (API %d) providers=%v", info.Version, info.APIVersion, info.Providers)
}
