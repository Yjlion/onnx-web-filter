package sitecat

import "testing"

func TestSiteKey(t *testing.T) {
	for in, want := range map[string]string{
		"www.amazon.co.uk":             "amazon.co.uk",
		"https://m.facebook.com/x?y=1": "facebook.com",
		"bank.example.com:443":         "example.com",
		"10.0.0.1":                     "10.0.0.1",
		"Example.COM.":                 "example.com",
	} {
		if got := SiteKey(in); got != want {
			t.Errorf("SiteKey(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPageKey(t *testing.T) {
	for in, want := range map[string]string{
		"https://Example.com":                           "example.com/",
		"http://example.com:8080/a/b?x=1#frag":          "example.com/a/b?x=1",
		"example.com/news":                              "example.com/news",
		"https://example.com/p?b=2&utm_source=x&a=1":    "example.com/p?a=1&b=2",
		"https://example.com/p?fbclid=abc&UTM_Medium=y": "example.com/p",
		"https://example.com/caf%C3%A9":                 "example.com/caf%C3%A9",
		"":                                              "",
		"http://":                                       "",
	} {
		if got := PageKey(in); got != want {
			t.Errorf("PageKey(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestNormalize(t *testing.T) {
	for in, want := range map[string]string{
		"Social Media": "social_media", "banking": "banking_finance", "news": "news",
		"Banking & Finance": "banking_finance", "nonsense": "",
	} {
		if got := Normalize(in); got != want {
			t.Errorf("Normalize(%q) = %q, want %q", in, got, want)
		}
	}
}

type fakeLists map[string][]string

func (f fakeLists) HostMatches(host, name string) bool {
	for _, h := range f[name] {
		if h == host {
			return true
		}
	}
	return false
}

func TestFromLists(t *testing.T) {
	lists := fakeLists{"shopping": {"shop.example"}, "social": {"both.example"}, "porn": {"both.example"}}
	if got := FromLists(lists, "shop.example"); got != "shopping" {
		t.Errorf("shop = %q", got)
	}
	if got := FromLists(lists, "both.example"); got != "adult" {
		t.Errorf("most restrictive list should win, got %q", got)
	}
	if got := FromLists(lists, "unknown.example"); got != "" {
		t.Errorf("unknown = %q", got)
	}
	for name, slug := range ListMap {
		if !Valid(slug) {
			t.Errorf("ListMap[%s] = %q is not a valid slug", name, slug)
		}
	}
	if len(listOrder) != len(ListMap) {
		t.Errorf("listOrder and ListMap out of sync")
	}
}
