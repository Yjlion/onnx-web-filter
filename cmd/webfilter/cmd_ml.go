package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"runtime"
	"text/tabwriter"

	"github.com/spf13/cobra"

	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortenv"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
)

// mlFlags locate the ONNX Runtime install. Until the ml settings block
// replaces the llm one, they are command-line only.
type mlFlags struct {
	dataDir string
	accel   string
	lib     string
}

func addMLFlags(cmd *cobra.Command) *mlFlags {
	f := &mlFlags{}
	cmd.Flags().StringVar(&f.dataDir, "data-dir", "./data/ml", "where the runtime and models are downloaded")
	cmd.Flags().StringVar(&f.accel, "accel", "auto", "runtime build: auto, cpu, cuda (CUDA 12) or cuda13")
	cmd.Flags().StringVar(&f.lib, "lib", "", "use this ONNX Runtime library (file or directory) instead of a download; ORT_LIB_PATH works too")
	return f
}

func newMLCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "ml",
		Short: "Manage the ONNX Runtime and classification models",
	}

	status := &cobra.Command{Use: "status", Short: "Load the ONNX Runtime and show its version and execution providers"}
	sf := addMLFlags(status)
	var asJSON bool
	status.Flags().BoolVar(&asJSON, "json", false, "print the status as JSON")
	status.RunE = func(cmd *cobra.Command, args []string) error {
		accel := ortrt.ResolveAccel(sf.accel)
		lib, err := ortenv.Locate(sf.lib, sf.dataDir, accel)
		if errors.Is(err, ortenv.ErrNotInstalled) {
			fmt.Printf("ONNX Runtime %s (%s) is not installed under %s\n", ortrt.PinnedVersion, accel, sf.dataDir)
			fmt.Println("run `webfilter ml download` to fetch it, or point --lib / ORT_LIB_PATH at an existing library")
			return nil
		}
		if err != nil {
			return err
		}
		info, err := ortenv.Init(lib)
		if err != nil {
			return err
		}
		defer ortenv.Shutdown()
		if asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(info)
		}
		fmt.Printf("library:    %s (%s)\n", info.Library.Path, info.Library.Source)
		if info.Library.Accel != "" {
			fmt.Printf("build:      %s\n", info.Library.Accel)
		}
		fmt.Printf("version:    %s (C API %d)\n", info.Version, info.APIVersion)
		fmt.Printf("providers:  %v\n", info.Providers)
		if accel.GPU() && !info.HasProvider("CUDAExecutionProvider") {
			fmt.Printf("\nthis machine looks CUDA-capable but the loaded build is CPU-only;\nrun `webfilter ml download --accel %s` for the GPU build\n", accel)
		}
		return nil
	}

	models := &cobra.Command{Use: "models", Short: "List the model catalog"}
	mf := addMLFlags(models)
	models.RunE = func(cmd *cobra.Command, args []string) error {
		tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "ID\tTASK\tSIZE\tDEFAULT\tINSTALLED\tLICENSE\tNAME")
		for _, m := range catalog.All() {
			_, inst := catalog.LoadInstalled(mf.dataDir, m)
			fmt.Fprintf(tw, "%s\t%s\t%d MB\t%v\t%v\t%s\t%s\n", m.ID, m.Task, m.SizeBytes()>>20, m.Default, inst, m.License, m.Name)
		}
		return tw.Flush()
	}

	download := &cobra.Command{
		Use:   "download [model-id...]",
		Short: "Download the ONNX Runtime " + ortrt.PinnedVersion + " build for this machine and models (default: one per task)",
	}
	df := addMLFlags(download)
	download.RunE = func(cmd *cobra.Command, args []string) error {
		var want []catalog.Model
		for _, id := range args {
			m, ok := catalog.Lookup(id)
			if !ok {
				return fmt.Errorf("unknown model %q; see `webfilter ml models`", id)
			}
			want = append(want, m)
		}
		if len(want) == 0 {
			for _, t := range []catalog.Task{catalog.TaskImage, catalog.TaskText, catalog.TaskSite} {
				want = append(want, catalog.Default(t))
			}
		}
		accel := ortrt.ResolveAccel(df.accel)
		if _, ok := ortrt.AssetFor(ortrt.PinnedVersion, runtime.GOOS, runtime.GOARCH, accel); !ok && accel.GPU() {
			fmt.Printf("no %s build for %s/%s; using the CPU build\n", accel, runtime.GOOS, runtime.GOARCH)
			accel = ortrt.AccelCPU
		}
		inst, err := ortrt.Download(cmd.Context(), df.dataDir, ortrt.PinnedVersion, accel, func(s string) { fmt.Println(s) })
		if err != nil {
			return err
		}
		fmt.Printf("installed %s\n  %s\n", inst.Asset, inst.Library)

		hub := catalog.NewHub()
		for _, m := range want {
			if _, ok := catalog.LoadInstalled(df.dataDir, m); ok {
				fmt.Printf("%s: already installed\n", m.ID)
				continue
			}
			fmt.Printf("%s: downloading %d MB from %s\n", m.ID, m.SizeBytes()>>20, m.Repo)
			if _, err := hub.Download(cmd.Context(), df.dataDir, m, nil); err != nil {
				return err
			}
			fmt.Printf("%s: installed\n", m.ID)
		}
		return nil
	}

	root.AddCommand(status, models, download)
	return root
}
