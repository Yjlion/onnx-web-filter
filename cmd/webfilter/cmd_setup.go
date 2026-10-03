package main

import (
	"bufio"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/yjlion/onnx-web-filter/internal/config"
	"github.com/yjlion/onnx-web-filter/internal/ml"
	"github.com/yjlion/onnx-web-filter/internal/ml/ortrt"
)

// newSetupCmd is the first-run wizard: create the settings and default
// policy, download the ONNX Runtime and the classification models, and
// print how to point browsers at the proxy.
func newSetupCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "setup",
		Short: "First-run setup: create config and download the runtime and models",
	}
	f := addConfigFlags(cmd)
	var yes, skipDownload bool
	cmd.Flags().BoolVarP(&yes, "yes", "y", false, "do not ask; accept defaults")
	cmd.Flags().BoolVar(&skipDownload, "skip-download", false, "only write config files")
	cmd.RunE = func(cmd *cobra.Command, args []string) error {
		if err := config.BootstrapRuntimeFiles(f.settingsPath); err != nil {
			return err
		}
		settings, err := config.LoadSettings(f.settingsPath)
		if err != nil {
			return err
		}
		cfg, err := loadMLConfig(f.settingsPath)
		if err != nil {
			return err
		}
		svc := ml.New(cfg)
		fmt.Printf("settings:   %s\n", f.settingsPath)
		fmt.Printf("policies:   %s\n", settings.PoliciesDir)
		fmt.Printf("ml data:    %s\n", cfg.DataDir)
		fmt.Printf("runtime:    ONNX Runtime %s (%s build)\n", ortrt.PinnedVersion, svc.Accel())
		var total int64
		for _, t := range ml.Tasks {
			m := svc.Model(t)
			total += m.SizeBytes()
			fmt.Printf("%-11s %s, %d MB (%s)\n", string(t)+" model:", m.ID, m.SizeBytes()>>20, m.License)
		}
		fmt.Println()
		if skipDownload {
			return nil
		}
		if rt, missing := svc.Missing(); !rt && len(missing) == 0 {
			fmt.Println("runtime and models already installed")
		} else {
			if !yes {
				fmt.Printf("Download the runtime and the models (~%d MB)? [Y/n] ", total>>20)
				line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
				if l := strings.ToLower(strings.TrimSpace(line)); l != "" && l != "y" && l != "yes" {
					fmt.Println("skipped; run `webfilter ml download` later")
					return nil
				}
			}
			ctx, stop := signal.NotifyContext(cmd.Context(), os.Interrupt, syscall.SIGTERM)
			defer stop()
			if err := downloadAndWait(ctx, cfg, ""); err != nil {
				return err
			}
		}
		fmt.Println()
		fmt.Println("Done. Start the filter with:")
		fmt.Printf("  webfilter run --settings %s\n", f.settingsPath)
		fmt.Println("then open the management UI, download the CA certificate from the Settings page and")
		fmt.Println("install it on each device, and point devices at the proxy (or use the PAC file).")
		return nil
	}
	return cmd
}
