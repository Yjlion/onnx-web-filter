package mgmtapi

import (
	"net/http/httptest"
	"testing"
)

func TestDecisionPathKeyKeepsPageKeys(t *testing.T) {
	for raw, want := range map[string][2]string{
		"/api/decisions/text/abc123":                                {"text", "abc123"},
		"/api/decisions/page_category/a.example%2Fnews%3Fa%3D1":     {"page_category", "a.example/news?a=1"},
		"/api/decisions/page_category/a.example%2Fcaf%25C3%25A9":    {"page_category", "a.example/caf%C3%A9"},
		"/api/decisions/page_category/a.example%2FCase%2FSensitive": {"page_category", "a.example/Case/Sensitive"},
	} {
		// Parse as a server would, then split from the escaped path.
		r := httptest.NewRequest("DELETE", raw, nil)
		kind, key, err := decisionPathKey(r.URL.EscapedPath())
		if err != nil || kind != want[0] || key != want[1] {
			t.Errorf("%s -> %q %q %v, want %q", raw, kind, key, err, want)
		}
	}
	if _, _, err := decisionPathKey("/api/decisions/text"); err == nil {
		t.Error("missing key accepted")
	}
}
