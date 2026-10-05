package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	"github.com/yjlion/onnx-web-filter/internal/config"
	"github.com/yjlion/onnx-web-filter/internal/ml"
	"github.com/yjlion/onnx-web-filter/internal/ml/catalog"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortenv"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
	"github.com/yjlion/onnx-web-filter/internal/models"
)

func newMLCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "ml",
		Short: "Manage the ONNX Runtime and the classification models",
	}

	status := &cobra.Command{Use: "status", Short: "Show the runtime, models and download state"}
	sf := addConfigFlags(status)
	var asJSON bool
	status.Flags().BoolVar(&asJSON, "json", false, "print the raw status JSON")
	status.RunE = func(cmd *cobra.Command, args []string) error {
		cfg, err := loadMLConfig(sf.settingsPath)
		if err != nil {
			return err
		}
		svc := ml.New(cfg)
		st := svc.Status()
		// Loading the library is cheap and tells which providers it has.
		var info *ortenv.Info
		if lib, err := ortenv.Locate(cfg.ORTLibPath, cfg.DataDir, svc.Accel()); err == nil {
			if i, err := ortenv.Init(lib); err == nil {
				info = &i
			} else {
				st.LastError = err.Error()
			}
		}
		if asJSON {
			enc := json.NewEncoder(os.Stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(struct {
				ml.Status
				Loaded *ortenv.Info `json:"library,omitempty"`
			}{st, info})
		}
		fmt.Printf("enabled:    %v\n", st.Enabled)
		fmt.Printf("data dir:   %s\n", st.DataDir)
		fmt.Printf("accel:      %s (parallel %d, %d threads each)\n", st.Accel, st.Slots, st.Threads)
		if info != nil {
			fmt.Printf("runtime:    ONNX Runtime %s from %s (%s)\n", info.Version, info.Library.Path, info.Library.Source)
			fmt.Printf("providers:  %s\n", strings.Join(info.Providers, ", "))
		} else {
			fmt.Printf("runtime:    ONNX Runtime %s installed=%v\n", ortrt.PinnedVersion, st.Runtime.Installed)
		}
		fmt.Println("models:")
		for _, m := range st.Models {
			fmt.Printf("  %-6s %-18s %4d MB  installed=%v\n", m.Task, m.ID, m.SizeMB, m.Installed)
		}
		if st.LastError != "" {
			fmt.Printf("error:      %s\n", st.LastError)
		}
		if rt, missing := svc.Missing(); rt || len(missing) > 0 {
			fmt.Println("\nrun `webfilter ml download` to fetch what is missing")
		}
		return nil
	}

	modelsCmd := &cobra.Command{Use: "models", Short: "List the model catalog"}
	mf := addConfigFlags(modelsCmd)
	modelsCmd.RunE = func(cmd *cobra.Command, args []string) error {
		cfg, err := loadMLConfig(mf.settingsPath)
		if err != nil {
			return err
		}
		svc := ml.New(cfg)
		tw := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "ID\tTASK\tSIZE\tCONFIGURED\tINSTALLED\tLICENSE\tNAME")
		for _, m := range catalog.All() {
			_, inst := catalog.LoadInstalled(cfg.DataDir, m)
			fmt.Fprintf(tw, "%s\t%s\t%d MB\t%v\t%v\t%s\t%s\n", m.ID, m.Task, m.SizeBytes()>>20,
				svc.Model(m.Task).ID == m.ID, inst, m.License, m.Name)
		}
		return tw.Flush()
	}

	download := &cobra.Command{
		Use:   "download [model-id]",
		Short: "Download the ONNX Runtime " + ortrt.PinnedVersion + " build for this machine and a model (default: every configured model)",
		Args:  cobra.MaximumNArgs(1),
	}
	df := addConfigFlags(download)
	download.RunE = func(cmd *cobra.Command, args []string) error {
		cfg, err := loadMLConfig(df.settingsPath)
		if err != nil {
			return err
		}
		id := ""
		if len(args) == 1 {
			id = args[0]
		}
		ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		return downloadAndWait(ctx, cfg, id)
	}

	remove := &cobra.Command{Use: "remove <model-id>", Short: "Delete an installed model that is not configured", Args: cobra.ExactArgs(1)}
	rf := addConfigFlags(remove)
	remove.RunE = func(cmd *cobra.Command, args []string) error {
		cfg, err := loadMLConfig(rf.settingsPath)
		if err != nil {
			return err
		}
		return ml.New(cfg).RemoveModel(args[0])
	}

	root.AddCommand(status, modelsCmd, download, remove, newMLTestCmd())
	return root
}

// newMLTestCmd classifies one input with the configured models, bypassing
// the decision cache: a quick way to see what the filter would decide.
func newMLTestCmd() *cobra.Command {
	test := &cobra.Command{Use: "test", Short: "Classify an image file, some text or a site with the configured models"}
	load := func(settingsPath string) (*ml.Service, error) {
		cfg, err := loadMLConfig(settingsPath)
		if err != nil {
			return nil, err
		}
		cfg.Enabled = true
		svc := ml.New(cfg)
		if err := svc.Start(context.Background()); err != nil {
			return nil, err
		}
		if !svc.Ready() {
			return nil, fmt.Errorf("runtime or models missing; run `webfilter ml download` first")
		}
		return svc, nil
	}
	printJSON := func(v any) error {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(v)
	}

	image := &cobra.Command{Use: "image <file>", Short: "Score an image file", Args: cobra.ExactArgs(1)}
	imf := addConfigFlags(image)
	image.RunE = func(cmd *cobra.Command, args []string) error {
		data, err := os.ReadFile(args[0])
		if err != nil {
			return err
		}
		svc, err := load(imf.settingsPath)
		if err != nil {
			return err
		}
		defer svc.Stop()
		started := time.Now()
		sc, err := svc.Image(cmd.Context(), data)
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s in %d ms\n", svc.Model(catalog.TaskImage).ID, time.Since(started).Milliseconds())
		return printJSON(sc)
	}

	text := &cobra.Command{Use: "text <text...>", Short: "Score text for adult content", Args: cobra.MinimumNArgs(1)}
	tf := addConfigFlags(text)
	text.RunE = func(cmd *cobra.Command, args []string) error {
		svc, err := load(tf.settingsPath)
		if err != nil {
			return err
		}
		defer svc.Stop()
		started := time.Now()
		sc, err := svc.Text(cmd.Context(), strings.Join(args, " "))
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s in %d ms\n", svc.Model(catalog.TaskText).ID, time.Since(started).Milliseconds())
		return printJSON(sc)
	}

	site := &cobra.Command{Use: "site <host> [title] [description]", Short: "Categorize a site", Args: cobra.RangeArgs(1, 3)}
	stf := addConfigFlags(site)
	site.RunE = func(cmd *cobra.Command, args []string) error {
		svc, err := load(stf.settingsPath)
		if err != nil {
			return err
		}
		defer svc.Stop()
		args = append(args, "", "")
		started := time.Now()
		sc, err := svc.Site(cmd.Context(), args[0], args[1], args[2])
		if err != nil {
			return err
		}
		fmt.Fprintf(os.Stderr, "%s in %d ms\n", svc.Model(catalog.TaskSite).ID, time.Since(started).Milliseconds())
		return printJSON(sc)
	}

	test.AddCommand(image, text, site)
	return test
}

// loadMLConfig reads the ml settings block, filling in the bootstrap data
// dir when the file leaves it empty.
func loadMLConfig(settingsPath string) (models.MLConfig, error) {
	if err := config.BootstrapRuntimeFiles(settingsPath); err != nil {
		return models.MLConfig{}, err
	}
	settings, err := config.LoadSettings(settingsPath)
	if err != nil {
		return models.MLConfig{}, fmt.Errorf("load settings: %w", err)
	}
	cfg := settings.ML
	if strings.TrimSpace(cfg.DataDir) == "" {
		cfg.DataDir = config.NewBootstrapSettings(settingsPath).ML.DataDir
	}
	return cfg, nil
}

// downloadAndWait downloads in the foreground, printing progress. The
// service is built disabled so it does not load the models afterwards.
func downloadAndWait(ctx context.Context, cfg models.MLConfig, modelID string) error {
	cfg.Enabled = false
	svc := ml.New(cfg)
	if a := svc.Accel(); a.GPU() {
		if _, ok := ortrt.AssetFor(ortrt.PinnedVersion, runtime.GOOS, runtime.GOARCH, a); !ok {
			fmt.Printf("no %s build of ONNX Runtime for %s/%s; the CPU build will be used\n", a, runtime.GOOS, runtime.GOARCH)
		}
	}
	if err := svc.Download(ctx, modelID); err != nil {
		return err
	}
	last := ""
	for {
		d := svc.Status().Download
		line := d.Step
		if d.BytesTot > 0 {
			line = fmt.Sprintf("%s  %5.1f%%  %d/%d MB", d.Step, d.Percent, d.BytesDone>>20, d.BytesTot>>20)
		}
		if line != last {
			fmt.Printf("\r\033[K%s", line)
			last = line
		}
		if !d.Active {
			fmt.Println()
			if d.Error != "" {
				return fmt.Errorf("download failed: %s", d.Error)
			}
			fmt.Printf("done in %.0fs\n", d.Seconds)
			return nil
		}
		select {
		case <-ctx.Done():
			svc.CancelDownload()
			fmt.Println("\ncancelled")
			return ctx.Err()
		case <-time.After(300 * time.Millisecond):
		}
	}
}
