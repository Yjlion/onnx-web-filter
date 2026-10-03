# Installing onnx-web-filter

onnx-web-filter is a single binary. It downloads two things on first run:
Microsoft's prebuilt [ONNX Runtime](https://github.com/microsoft/onnxruntime)
library for your platform (about 11 MB compressed for the CPU build, more
for CUDA) and three small ONNX models (about 105 MB together). Nothing else
is installed.

## Requirements

| | Minimum | Comfortable |
|---|---|---|
| RAM | 1 GB free | 2 GB+ |
| CPU | x86-64 or 64-bit ARM | 4 cores; an NVIDIA GPU is supported but rarely needed |
| Disk | 300 MB in the data directory | |
| OS | Windows 10/11 x64, macOS 13+ on Apple silicon, Linux x64/ARM64 (glibc 2.31+, e.g. Ubuntu 20.04, Debian 11) | |

## 1. Download

Grab the archive for your platform from the Releases page and unpack it
anywhere, for example `C:\webfilter`, `~/webfilter` or `/opt/webfilter`.

| Platform | Archive |
|---|---|
| Windows x64 | `webfilter-<version>-windows-amd64.zip` |
| macOS Apple silicon | `webfilter-<version>-darwin-arm64.tar.gz` |
| Linux x64 / ARM64 | `webfilter-<version>-linux-amd64.tar.gz` / `-arm64.tar.gz` |

Or build from source with Go 1.26+ and a C compiler (gcc or clang; MinGW
gcc on Windows), which the ONNX Runtime Go bindings need:

```sh
go build -o webfilter ./cmd/webfilter
```

## 2. Set up

```sh
./webfilter setup
```

The wizard creates `config/settings.json` and `policies/default.json` next to
the binary and downloads ONNX Runtime and the models into `data/ml/`. With
an NVIDIA driver present on x64 Linux or Windows it fetches the CUDA build
of the runtime. `--yes` accepts the defaults, `--skip-download` only writes
the config.

The same thing is available later as `webfilter ml download` and from the
**Models** page of the management UI. `webfilter ml test image photo.jpg`
shows what the image model makes of a picture.

## 3. Run

```sh
./webfilter run
```

| Address | What |
|---|---|
| `http://127.0.0.1:8000` | Management UI and API |
| `127.0.0.1:8080` | HTTP/HTTPS forward proxy |
| `127.0.0.1:1080` | SOCKS5 proxy |

To serve other devices on the LAN, set `proxy_listen` and `mgmt_host` to
`0.0.0.0` in Settings (or in `config/settings.json`) and restart.

## 4. Trust the CA and point devices at the proxy

The proxy decrypts HTTPS to inspect pages and images, so each client must
trust its certificate authority. Download it from **Settings → Certificates**
(or `http://<proxy>:8000/api/ca-cert`) and install it as a trusted root on
every device. Then set the device's proxy to the filter, or hand out the PAC
file at `http://<proxy>:8000/proxy.pac`.

Sites listed under a policy's **MITM Control → exclude** (banking, by default
`chase.com` in the example policy) are tunnelled without inspection.

## 5. Turn on filtering

Nothing is filtered until a policy says so. On the **Policies** page, open
the default policy and switch on the text and image classifiers, the site
categories you want blocked and ad blocking. See [policies.md](policies.md).

## Running as a service

**Linux (systemd)** — create `/etc/systemd/system/webfilter.service`:

```ini
[Unit]
Description=onnx-web-filter
After=network-online.target

[Service]
ExecStart=/opt/webfilter/webfilter run --settings /opt/webfilter/config/settings.json
WorkingDirectory=/opt/webfilter
Restart=on-failure
User=webfilter

[Install]
WantedBy=multi-user.target
```

**Windows** — `webfilter service install` registers a Windows service that
runs `webfilter run`; `service uninstall` removes it.

**macOS** — use a `launchd` plist running the same `run` command, or start it
from a terminal.

**Docker** — see [docker.md](docker.md).

## Updating

Replace the binary. Settings, policies, rules, certificates, logs and the
downloaded models are all outside it. When a release pins a newer ONNX
Runtime or model, `webfilter ml download` fetches what changed.

## Where things live

```
config/settings.json     global settings
config/rules.json        named devices (and sentence rules from earlier versions)
policies/*.json          per-client policies
certs/                   the CA and issued certificates
logs/webfilter.db        request and block log (SQLite)
data/ml/                 ONNX Runtime, models, decisions.db (the verdict cache)
data/adblock/            downloaded filter lists (empty = built-in snapshot)
categories/              site-category blocklists (optional)
```
