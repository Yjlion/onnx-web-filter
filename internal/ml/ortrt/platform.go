// Package ortrt downloads a prebuilt ONNX Runtime release for this machine
// from github.com/microsoft/onnxruntime/releases and locates its shared
// library. It does not load the library; see package ortenv.
package ortrt

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// PinnedVersion is the ONNX Runtime release the Go bindings and the model
// catalog are tested against. The bindings negotiate C API versions 27 down
// to 17, so a library supplied by hand (ORT_LIB_PATH) loads from 1.17 on.
const PinnedVersion = "1.30.0"

// ReleaseBase is where release assets are fetched from. ORT_RELEASE_BASE
// overrides it (mirrors, tests).
const ReleaseBase = "https://github.com/microsoft/onnxruntime/releases/download"

// Accel is an ONNX Runtime build flavour.
type Accel string

const (
	AccelCPU Accel = "cpu"
	// AccelCUDA is the CUDA 12 build: it runs on any driver from the 525
	// series on, which covers more machines than the CUDA 13 build.
	AccelCUDA   Accel = "cuda"
	AccelCUDA13 Accel = "cuda13"
)

// GPU reports whether the flavour carries the CUDA execution provider.
func (a Accel) GPU() bool { return a == AccelCUDA || a == AccelCUDA13 }

// Asset is one downloadable release archive.
type Asset struct {
	Name string
	URL  string
}

// AssetFor maps (OS, arch, accel) to the release archive name. ok=false means
// no prebuilt exists for the combination (1.30.0 ships no macOS x64 build,
// and CUDA builds only for x64).
func AssetFor(version, goos, goarch string, accel Accel) (Asset, bool) {
	var plat, ext string
	switch goos + "/" + goarch {
	case "linux/amd64":
		plat, ext = "linux-x64", ".tgz"
	case "linux/arm64":
		plat, ext = "linux-aarch64", ".tgz"
	case "darwin/arm64":
		plat, ext = "osx-arm64", ".tgz"
	case "windows/amd64":
		plat, ext = "win-x64", ".zip"
	case "windows/arm64":
		plat, ext = "win-arm64", ".zip"
	default:
		return Asset{}, false
	}
	switch accel {
	case AccelCPU:
	case AccelCUDA, AccelCUDA13:
		if goarch != "amd64" || goos == "darwin" {
			return Asset{}, false
		}
		if accel == AccelCUDA {
			plat += "-gpu_cuda12"
		} else {
			plat += "-gpu_cuda13"
		}
	default:
		return Asset{}, false
	}
	name := "onnxruntime-" + plat + "-" + version + ext
	return Asset{Name: name, URL: releaseBase() + "/v" + version + "/" + name}, true
}

func releaseBase() string {
	if v := strings.TrimRight(strings.TrimSpace(os.Getenv("ORT_RELEASE_BASE")), "/"); v != "" {
		return v
	}
	return ReleaseBase
}

// ResolveAccel turns the configured value ("auto", "cpu", "cuda", "cuda13")
// into the flavour to download, probing the machine for "auto".
func ResolveAccel(configured string) Accel {
	switch strings.ToLower(strings.TrimSpace(configured)) {
	case "cpu":
		return AccelCPU
	case "cuda", "cuda12":
		return AccelCUDA
	case "cuda13":
		return AccelCUDA13
	}
	return ProbeAccel()
}

// ProbeAccel guesses the best flavour for this machine. It only checks for
// the NVIDIA driver; whether CUDA and cuDNN are actually usable is found out
// when a session appends the CUDA execution provider, which falls back to
// the CPU provider on failure.
func ProbeAccel() Accel {
	if runtime.GOARCH != "amd64" {
		return AccelCPU
	}
	switch runtime.GOOS {
	case "windows":
		sys := filepath.Join(os.Getenv("SystemRoot"), "System32")
		if sys == "System32" {
			sys = `C:\Windows\System32`
		}
		if exists(filepath.Join(sys, "nvcuda.dll")) || lookPath("nvidia-smi") {
			return AccelCUDA
		}
	case "linux":
		if lookPath("nvidia-smi") || exists("/dev/nvidiactl") {
			return AccelCUDA
		}
	}
	return AccelCPU
}

func exists(p string) bool   { _, err := os.Stat(p); return err == nil }
func lookPath(n string) bool { _, err := exec.LookPath(n); return err == nil }

// LibraryName is the ONNX Runtime shared library name on this OS.
func LibraryName() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	}
	return "libonnxruntime.so"
}

// InstallDir is where a runtime flavour is unpacked.
func InstallDir(dataDir, version string, accel Accel) string {
	return filepath.Join(dataDir, "runtime", fmt.Sprintf("%s-%s-%s-%s", version, runtime.GOOS, runtime.GOARCH, accel))
}

// FindLibrary locates the shared library inside an install dir (archives
// nest it under onnxruntime-<plat>-<version>/lib/).
func FindLibrary(installDir string) (string, bool) {
	want := LibraryName()
	var found string
	_ = filepath.WalkDir(installDir, func(p string, d os.DirEntry, err error) error {
		if err != nil || found != "" {
			return nil
		}
		if d.Name() == want && !d.IsDir() {
			found = p
			return filepath.SkipAll
		}
		return nil
	})
	return found, found != ""
}
