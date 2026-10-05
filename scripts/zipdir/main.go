// Command zipdir writes a zip archive of a directory with forward-slash entry
// names, for scripts/package-release.sh on Windows. PowerShell's
// Compress-Archive writes backslash separators, which unzip and other tools
// mishandle, and Git Bash ships no zip.
//
// Usage: go run ./scripts/zipdir <dir> <out.zip>
package main

import (
	"archive/zip"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: zipdir <dir> <out.zip>")
		os.Exit(2)
	}
	if err := zipDir(os.Args[1], os.Args[2]); err != nil {
		fmt.Fprintln(os.Stderr, "zipdir:", err)
		os.Exit(1)
	}
}

// zipDir stores dir under its own base name, as `zip -r out.zip dir` does.
func zipDir(dir, out string) error {
	f, err := os.Create(out)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	parent := filepath.Dir(filepath.Clean(dir))
	err = filepath.WalkDir(dir, func(p string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(parent, p)
		if err != nil {
			return err
		}
		name := filepath.ToSlash(rel)
		info, err := d.Info()
		if err != nil {
			return err
		}
		h, err := zip.FileInfoHeader(info)
		if err != nil {
			return err
		}
		h.Name = name
		if d.IsDir() {
			h.Name += "/"
			_, err = zw.CreateHeader(h)
			return err
		}
		h.Method = zip.Deflate
		w, err := zw.CreateHeader(h)
		if err != nil {
			return err
		}
		src, err := os.Open(p)
		if err != nil {
			return err
		}
		defer src.Close()
		_, err = io.Copy(w, src)
		return err
	})
	if cerr := zw.Close(); err == nil {
		err = cerr
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}
