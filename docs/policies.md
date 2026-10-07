# Policies and site categories

**Policies** (`policies/*.json`) are per-client configurations matched by
MAC, exact IP, CIDR or catch-all, inherited from gowebfilter: URL allow and
block lists, site categories, SafeSearch, YouTube channel filtering,
DNS-over-HTTPS blocking, MITM control, the text, image and video classifiers
and ad blocking. A policy with a schedule applies only inside its time
windows and, while active, takes priority over an unscheduled policy for
the same devices. Edit policies on the **Policies** page.

llama-web-filter, which this project is forked from, also had a
natural-language assistant that proposed policy changes. It needs a chat
model, so it is not part of onnx-web-filter.

### Rules from earlier versions

Earlier versions of llama-web-filter turned one sentence into an overlay
rule in `config/rules.json`. Those rules are still enforced, so nothing
loosens when a configuration is carried over, but new ones cannot be
added. `webfilter rules list` shows them and `webfilter rules remove <id>`
deletes one.

## Site categories

Every website gets one category from a fixed list:

adult, dating, gambling, social media, chat & messaging, email, news,
shopping, banking & finance, video streaming, music & audio, games,
entertainment, sports, search engines, education, kids, government,
health, travel, jobs, religion, technology, business, ads & tracking,
malware & phishing, piracy, drugs & alcohol, weapons, violence & hate,
infrastructure (CDNs, APIs, login and asset servers), other.

```json
"category_filter": { "enabled": true, "mode": "blacklist", "categories": ["shopping", "social_media"],
                     "on_timeout": "allow", "on_unavailable": "allow", "budget_ms": 0 }
```

* `mode` `blacklist` blocks the listed categories; `whitelist` allows only
  them. Infrastructure hosts are always allowed in `whitelist` mode, and so
  are the resources an allowed page loads, or nothing would display.
* Where a category comes from: a manual override on the **Decisions** page,
  then the installed domain lists (`webfilter categories update`; porn,
  gambling, shopping, social, streaming and the other IPFire lists map onto
  the categories above), then the model's cached verdict, then the model.
  Site verdicts are cached per registrable domain.
* **Every page is categorized too.** When a navigation's HTML arrives, the
  model reads the page itself (URL path, title, description, headings and
  the start of its text) and the page's category wins over its site's, lists
  included: a news article on a shopping site is news, and is blocked if
  news is. A site whose own category is blocked is still refused before
  anything is fetched; page verdicts only add blocks on allowed sites.
  Page verdicts are cached per URL (host, path and query, minus tracking
  parameters such as `utm_*` and `fbclid`) together with a hash of the
  content they were judged from. Every load of a page is checked against
  it: unchanged content is a cache hit, changed content is judged again on
  that same load, so a blocked page that changes to an allowed category is
  let through. If the model does not answer within the budget the site's
  category stands for that load. Page verdicts not refreshed for
  `ml.page_category_days` (30) days are dropped. A manual override for the
  page (applied before the fetch), its host or its site wins over the page
  verdict.
* Only a page navigation asks the model and waits for it, up to
  `budget_ms` (or `ml.budget.category_ms`, 500 ms). If it has not decided
  by then the page gets `on_timeout` and the category is ready for the next
  visit. `on_unavailable` applies when the models are not loaded. How the
  embedding model picks a category is described in [ml.md](ml.md).
* Sub-resources (scripts, images, API calls) are judged only from what is
  already known, so one page's dozens of third-party hosts never queue model
  calls. A host already known to be in a blocked category is refused.
* Hosts that are tunnelled without inspection (MITM `exclude` sites such as
  banks) are checked from the lists and the cache when the connection
  opens, and categorized in the background for next time. Only
  `blacklist` mode is enforced there.
* The URL allow list wins over categories; the block list is checked first.
* The policy editor's **Site Categories** section has *Test a site*, which
  also takes a page URL; the **Decisions** page lists every cached category
  (filters *Site categories* and *Page categories*) and lets you set one by
  hand for a domain, an exact host or one page URL.

`url_filter.categories` still selects raw domain lists by name, as before.

## Classifier settings in a policy

```json
"text_classifier": { "enabled": true, "threshold": 0.8, "on_timeout": "allow", "on_unavailable": "allow", "budget_ms": 0,
                      "exclude": [], "include_only": [] },
"image_classifier": { "enabled": true, "action": "blur", "threshold": 0.75, "min_dimension": 100,
                      "on_timeout": "blur", "on_unavailable": "allow", "budget_ms": 0, "prefetch": true,
                      "exclude": [], "include_only": [] },
"video_classifier": { "enabled": true, "threshold": 0.75, "on_timeout": "allow", "keyframes": false, "youtube": true,
                      "exclude": [], "include_only": [] },
"adblock":          { "enabled": true, "cosmetic": true, "classify_unknown_hosts": true, "exclude": [], "include_only": [] }
```

* `threshold` is compared with the model's adult score (0–1). A score of
  `ml.adult_score` (0.9) or more is blocked whatever the threshold.
* `on_timeout` is what the browser gets when the verdict is not back within
  the budget on first sight: `allow` or `block` for pages; `allow`, `blur`,
  `checkerboard` or `block` for images. The verdict is cached either way.
* `on_unavailable` applies when no model can answer at all.
* `prefetch` scores a page's images before the browser asks for them.
* `adblock.cosmetic` injects element-hiding CSS; `classify_unknown_hosts`
  applies host decisions made by hand on the **Decisions** page to
  third-party hosts the lists do not know (no model judges hosts).
* `video_classifier` judges videos through stills: the poster a page
  declares for a `<video>`, a YouTube video's thumbnail when the player
  response passes through (`youtube`), and, with `keyframes` on and ffmpeg
  installed on the proxy machine, frames decoded from the stream's opening
  seconds. A video judged adult has its sources removed from the page, its
  YouTube player response made unplayable, and later requests for the stream
  refused.

## Ad blocking

Ad and tracker requests are matched against EasyList and EasyPrivacy. A
snapshot is built into the binary; **Settings → Ad blocking → Update** or
`webfilter adblock update` downloads the current lists into `data/adblock/`.
Blocked sub-resources get an empty response of the right type; navigating to
an ad host shows the block page. Blocked ad requests are counted in
`/metrics` (`webfilter_blocks_total{component="adblock"}`) and appear in the
request log, but are not written to the block log.

## Decisions

The **Decisions** page lists every cached verdict (images, pages, learned
sites, hosts) with its score, source and how often it was used. Mark any of
them clean or adult, add a site or host override, or clear the model's
verdicts. Manual overrides are never overwritten by the model.
