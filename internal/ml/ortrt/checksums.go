package ortrt

// pinnedSHA256 holds the SHA-256 GitHub publishes for each PinnedVersion
// release asset. A download of a pinned asset that does not match is
// rejected; assets of other versions (ORT_RELEASE_BASE mirrors, tests) are
// fetched unverified.
var pinnedSHA256 = map[string]string{
	"onnxruntime-linux-aarch64-1.30.0.tgz":        "e16a27a8ed330bbc698df7330b0cf56e722f354e3bcc92118682c74ef3c3e3da",
	"onnxruntime-linux-x64-1.30.0.tgz":            "a5ed5a3cac51fbb2e90da632ae43d19212faaa20e76484e62bcb7c23ddb3b3fd",
	"onnxruntime-linux-x64-gpu_cuda12-1.30.0.tgz": "f9886932ee7bb0b4d3fcab736a392d4ff5efaa0672b47f19f0cec03437cf64f1",
	"onnxruntime-linux-x64-gpu_cuda13-1.30.0.tgz": "382d79133112388cf94ce5855789b7c9bef12bef76a08b6b277e5a317213adcd",
	"onnxruntime-osx-arm64-1.30.0.tgz":            "6ebb5062a934537c352937821f9fe9718e7de1a2db1122a93dd363ffd53a7012",
	"onnxruntime-win-arm64-1.30.0.zip":            "e53db8a50b23ae35be901cc93428baf997dc8d420333b097b2eae53d3ea9f2d3",
	"onnxruntime-win-x64-1.30.0.zip":              "c6ba983baf5681af108599675d2a89c2d145512d02de28aed0bff177cd0ba949",
	"onnxruntime-win-x64-gpu_cuda12-1.30.0.zip":   "d4667ea48eb0a10bc9b96b838f7b8975a6bf18de3bc5edd403a22e15c1458b23",
	"onnxruntime-win-x64-gpu_cuda13-1.30.0.zip":   "8fa4b08359af682cd605892cb59077049700b640128bb93fd2c7776cf9f55bdc",
}
