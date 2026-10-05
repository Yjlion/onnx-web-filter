package sitecat

import "testing"

func TestEveryCategoryHasExamples(t *testing.T) {
	for _, c := range all {
		if len(Examples[c.Slug]) < 3 {
			t.Errorf("%s has %d examples, want at least 3", c.Slug, len(Examples[c.Slug]))
		}
	}
	for slug := range Examples {
		if !Valid(slug) {
			t.Errorf("examples for unknown slug %q", slug)
		}
	}
	protos := PrototypeTexts()
	if len(protos) != len(all) || protos[0].Texts[0] != all[0].Label+": "+all[0].Description {
		t.Errorf("PrototypeTexts = %+v", protos[0])
	}
}
