#!/usr/bin/env bash
# capture_screenshots.sh - render the management web UI to PNGs in screenshots/.
#
# Runs against a throwaway data directory seeded by scripts/seed_sample_data.go,
# so the repo's real config/, policies/ and logs/ are never touched. With the
# runtime and models available (--ml-data), it also loads them and sends a few
# sample pages and images through the proxy, so the Models and Decisions pages
# and the block page show real verdicts.
#
# Usage:
#   bash scripts/capture_screenshots.sh [--ml-data DIR] [--out DIR] [--port N] [--keep]
#
#   --ml-data DIR  a data dir filled by `webfilter ml download` (default:
#                  <repo>/data/ml if it holds a runtime; without one the
#                  model-dependent shots show "needs download")
#   --out DIR      where to write PNGs (default: <repo>/screenshots)
#   --port N       mgmt port for the temporary server (default: 8099; the
#                  proxy uses N-1 and the sample web server N-2)
#   --keep         don't delete the temporary data directory on exit
#
# Requires a Chromium/Chrome binary (CHROME=/path overrides the search),
# python3 for the sample web server, and curl.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$REPO_ROOT/screenshots"
PORT=8099
KEEP=0
ML_DATA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ml-data) ML_DATA="$2"; shift 2 ;;
    --out)     OUT_DIR="$2"; shift 2 ;;
    --port)    PORT="$2"; shift 2 ;;
    --keep)    KEEP=1; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
PROXY_PORT=$((PORT - 1))
WWW_PORT=$((PORT - 2))
if [[ -z "$ML_DATA" && -d "$REPO_ROOT/data/ml/runtime" ]]; then
  ML_DATA="$REPO_ROOT/data/ml"
fi

# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
if [[ -n "${CHROME:-}" ]]; then
  BROWSER="$CHROME"
else
  BROWSER=""
  for c in chromium chromium-browser google-chrome google-chrome-stable chrome; do
    if command -v "$c" >/dev/null 2>&1; then BROWSER="$(command -v "$c")"; break; fi
  done
fi
if [[ -z "$BROWSER" || ! -x "$BROWSER" ]]; then
  echo "error: no Chromium/Chrome binary found; set CHROME=/path/to/chromium" >&2
  exit 1
fi
for t in go curl python3; do
  command -v "$t" >/dev/null 2>&1 || { echo "error: $t not found in PATH" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
# build + seed + serve
# ---------------------------------------------------------------------------
DATA_DIR="$(mktemp -d "${TMPDIR:-/tmp}/webfilter-screenshots.XXXXXX")"
BIN="$DATA_DIR/webfilter"
SERVER_PID=""
WWW_PID=""

cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null || true
  [[ -n "$WWW_PID" ]] && kill "$WWW_PID" 2>/dev/null || true
  if [[ "$KEEP" -eq 1 ]]; then
    echo "[shots] kept data dir: $DATA_DIR"
  else
    rm -rf "$DATA_DIR"
  fi
}
trap cleanup EXIT

echo "[shots] building webfilter ..."
(cd "$REPO_ROOT" && go build -o "$BIN" ./cmd/webfilter)

echo "[shots] seeding sample data ..."
seed_args=(-dir "$DATA_DIR" -mgmt-port "$PORT" -proxy-port "$PROXY_PORT")
if [[ -n "$ML_DATA" ]]; then
  # Copy rather than point at it, so the Models page shows the throwaway
  # dir's paths and nothing about the machine the shots were taken on.
  mkdir -p "$DATA_DIR/data"
  cp -r "$ML_DATA" "$DATA_DIR/data/ml"
  rm -f "$DATA_DIR/data/ml/decisions.db"
fi
(cd "$REPO_ROOT" && go run scripts/seed_sample_data.go "${seed_args[@]}" >/dev/null)

echo "[shots] starting server on 127.0.0.1:$PORT (proxy :$PROXY_PORT) ..."
"$BIN" run --settings "$DATA_DIR/config/settings.json" >"$DATA_DIR/server.log" 2>&1 &
SERVER_PID=$!

API="http://127.0.0.1:$PORT"
for _ in $(seq 1 100); do
  curl -sf "$API/api/status" >/dev/null 2>&1 && break
  sleep 0.2
done
if ! curl -sf "$API/api/status" >/dev/null 2>&1; then
  echo "error: server did not come up; log follows" >&2
  cat "$DATA_DIR/server.log" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# sample traffic, so verdicts and the block page are real
# ---------------------------------------------------------------------------
WWW="$DATA_DIR/www"
mkdir -p "$WWW"
cat >"$WWW/library.html" <<'EOF'
<html><head><title>City library opening hours</title></head><body><h1>Welcome to the city library</h1>
<p>Opening hours, events for children, the new reading room and the summer book club. Borrow books,
ebooks and audiobooks with your library card. Our librarians help with research and homework every
afternoon.</p><img src="scene.jpg"></body></html>
EOF
cat >"$WWW/recipes.html" <<'EOF'
<html><head><title>Slow-roasted tomatoes</title></head><body><h1>Slow-roasted tomatoes with garlic</h1>
<p>Halve the tomatoes, drizzle with olive oil, scatter sliced garlic and thyme, and roast at a low
temperature for three hours until sweet and jammy. Serve on toast or stir through pasta with fresh
basil and a little parmesan.</p></body></html>
EOF
cat >"$WWW/cams.html" <<'EOF'
<html><head><title>Live cams</title></head><body><h1>Hot amateur webcam shows</h1>
<p>Watch naked girls stripping live, explicit hardcore videos updated daily. Join free tonight and chat
with horny models in private shows. Uncensored adult entertainment for members only.</p></body></html>
EOF
cp "$REPO_ROOT/internal/ml/classify/testdata/scene.jpg" "$WWW/"
python3 -m http.server "$WWW_PORT" --bind 127.0.0.1 -d "$WWW" >/dev/null 2>&1 &
WWW_PID=$!
sleep 0.5

if [[ -n "$ML_DATA" ]]; then
  echo "[shots] waiting for the models ..."
  for _ in $(seq 1 120); do
    curl -sf "$API/api/ml/status" | grep -q '"phase":"ready"' && break
    sleep 0.5
  done
  echo "[shots] sending sample traffic ..."
  for page in library.html recipes.html cams.html scene.jpg; do
    curl -s -o /dev/null -x "http://127.0.0.1:$PROXY_PORT" "http://127.0.0.1:$WWW_PORT/$page" || true
  done
  for host in github.com nytimes.com amazon.com wikipedia.org netflix.com bet365.com \
      espn.com booking.com indeed.com khanacademy.org tinder.com reddit.com; do
    curl -s -o /dev/null "$API/api/sitecategories/lookup?host=$host" || true
  done
fi
# The page a filtered client sees.
curl -s -x "http://127.0.0.1:$PROXY_PORT" -o "$DATA_DIR/block-page.html" "http://127.0.0.1:$WWW_PORT/cams.html" || true

# ---------------------------------------------------------------------------
# capture
# ---------------------------------------------------------------------------
mkdir -p "$OUT_DIR"

# shoot <name> <url> <width> <height> [extra chromium flags...]
shoot() {
  local name="$1" url="$2" width="$3" height="$4"
  shift 4
  echo "[shots]   $name.png  (${width}x${height})"
  "$BROWSER" \
    --headless --no-sandbox --disable-gpu --disable-dev-shm-usage \
    --hide-scrollbars --force-device-scale-factor=1 \
    --virtual-time-budget=10000 \
    --window-size="$width,$height" \
    --screenshot="$OUT_DIR/$name.png" \
    "$@" "$url" >/dev/null 2>&1
}

echo "[shots] capturing ..."
shoot index          "$API/index.html"                     1440 1000
shoot index-dark     "$API/index.html"                     1440 1000 --force-dark-mode
shoot policies       "$API/policies.html"                  1440  700
shoot policy-editor  "$API/policy-editor.html?name=kids"   1440 1400
shoot logs           "$API/logs.html"                      1440 1100
shoot analytics      "$API/analytics.html"                 1440 1250
shoot analytics-dark "$API/analytics.html"                 1440 1250 --force-dark-mode
shoot decisions      "$API/decisions.html"                 1440 1000
shoot decisions-dark "$API/decisions.html"                 1440 1000 --force-dark-mode
shoot models         "$API/models.html"                    1440 1300
shoot tools          "$API/tools.html"                     1440 1500
shoot settings       "$API/settings.html"                  1440 1600
# Served by a running filter with auth off, login.html redirects to the
# dashboard, so it is opened from the repo; its auth check fails and it stays.
shoot login          "file://$REPO_ROOT/ui/login.html"     1440  800
if grep -q "Access Blocked" "$DATA_DIR/block-page.html" 2>/dev/null; then
  shoot block-page "file://$DATA_DIR/block-page.html"      1440  800
else
  echo "[shots]   block-page.png skipped (no models, so nothing was blocked)"
fi

echo "[shots] done -> $OUT_DIR"
ls -1 "$OUT_DIR"
