// Package ortenv finds the ONNX Runtime shared library and loads it into the
// process through the official Go bindings (github.com/microsoft/onnxruntime/go).
// Loading happens once per process; the bindings share one OrtEnv between
// all sessions.
//
// The bindings use CGO, so this package (unlike ortrt) needs a C toolchain
// to build.
package ortenv

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	ort "github.com/microsoft/onnxruntime/go/onnxruntime"

	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
)

// Library is a located shared library and where it came from.
type Library struct {
	Path string `json:"path"`
	// Source is "config", "env" (ORT_LIB_PATH) or "download".
	Source string `json:"source"`
	// Accel is the downloaded flavour; empty for config/env libraries,
	// whose flavour is unknown until providers are listed.
	Accel ortrt.Accel `json:"accel,omitempty"`
}

// ErrNotInstalled means no library was configured and none is downloaded.
var ErrNotInstalled = errors.New("ONNX Runtime is not installed")

// Locate picks the library to load: an explicit path (file or directory)
// wins, then the ORT_LIB_PATH environment variable, then the downloaded
// runtime for accel under dataDir. A GPU flavour that is not downloaded
// falls back to a downloaded CPU runtime.
func Locate(explicit, dataDir string, accel ortrt.Accel) (Library, error) {
	if explicit = strings.TrimSpace(explicit); explicit != "" {
		p, err := resolve(explicit)
		return Library{Path: p, Source: "config"}, err
	}
	if env := strings.TrimSpace(os.Getenv("ORT_LIB_PATH")); env != "" {
		p, err := resolve(env)
		return Library{Path: p, Source: "env"}, err
	}
	flavours := []ortrt.Accel{accel}
	if accel != ortrt.AccelCPU {
		flavours = append(flavours, ortrt.AccelCPU)
	}
	for _, a := range flavours {
		if inst, ok := ortrt.LoadInstalled(dataDir, ortrt.PinnedVersion, a); ok {
			return Library{Path: inst.Library, Source: "download", Accel: a}, nil
		}
	}
	return Library{}, ErrNotInstalled
}

// resolve accepts the library file or the directory holding it.
func resolve(p string) (string, error) {
	info, err := os.Stat(p)
	if err != nil {
		return "", fmt.Errorf("ONNX Runtime library %s: %w", p, err)
	}
	if info.IsDir() {
		lib := filepath.Join(p, ortrt.LibraryName())
		if _, err := os.Stat(lib); err != nil {
			// Accept the root of an unpacked release too (lib/ inside).
			if found, ok := ortrt.FindLibrary(p); ok {
				return found, nil
			}
			return "", fmt.Errorf("ONNX Runtime library: no %s in %s", ortrt.LibraryName(), p)
		}
		p = lib
	}
	if abs, err := filepath.Abs(p); err == nil {
		p = abs
	}
	return p, nil
}

// Info describes the loaded runtime.
type Info struct {
	Library    Library  `json:"library"`
	Version    string   `json:"version"`
	APIVersion int      `json:"api_version"`
	Providers  []string `json:"providers"`
}

// HasProvider reports whether the runtime was built with the named
// execution provider (e.g. "CUDAExecutionProvider").
func (i Info) HasProvider(name string) bool {
	for _, p := range i.Providers {
		if p == name {
			return true
		}
	}
	return false
}

// Init loads lib and initialises the runtime. It is idempotent after the
// first success; a later call with a different library returns the info of
// the one already loaded, since a process can hold only one.
func Init(lib Library) (Info, error) {
	ort.SetSharedLibraryPath(lib.Path)
	if err := ort.Init(); err != nil {
		return Info{}, fmt.Errorf("load ONNX Runtime from %s: %w", lib.Path, err)
	}
	_ = ort.DisableTelemetry() // a no-op on Linux/macOS builds
	v, err := ort.GetVersion()
	if err != nil {
		return Info{}, err
	}
	providers, err := ort.AvailableProviders()
	if err != nil {
		return Info{}, err
	}
	return Info{Library: lib, Version: v, APIVersion: ort.APIVersion(), Providers: providers}, nil
}

// Shutdown releases the runtime. Sessions must be closed first.
func Shutdown() error { return ort.Shutdown() }
