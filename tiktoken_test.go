package tiktoken

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestEncoding(t *testing.T) {
	ass := assert.New(t)
	enc, err := EncodingForModel("gpt-3.5-turbo-16k")
	ass.Nil(err, "Encoding  init should not be nil")
	tokens := enc.Encode("hello world!你好，世界！", []string{"all"}, []string{"all"})
	// these tokens are converted from the original python code
	sourceTokens := []int{15339, 1917, 0, 57668, 53901, 3922, 3574, 244, 98220, 6447}
	ass.ElementsMatch(sourceTokens, tokens, "Encoding should be equal")

	tokens = enc.Encode("hello <|endoftext|>", []string{"<|endoftext|>"}, nil)
	sourceTokens = []int{15339, 220, 100257}
	ass.ElementsMatch(sourceTokens, tokens, "Encoding should be equal")

	tokens = enc.Encode("hello <|endoftext|>", []string{"<|endoftext|>"}, []string{"all"})
	sourceTokens = []int{15339, 220, 100257}
	ass.ElementsMatch(sourceTokens, tokens, "Encoding should be equal")

	ass.Panics(func() {
		tokens = enc.Encode("hello <|endoftext|><|endofprompt|>", []string{"<|endoftext|>"}, []string{"all"})
	})
	ass.Panics(func() {
		tokens = enc.Encode("hello <|endoftext|>", []string{"<|endoftext|>"}, []string{"<|endoftext|>"})
	})
}

func TestEncodingForModelMappings(t *testing.T) {
	testCases := map[string]string{
		"o1":                      MODEL_O200K_BASE,
		"o1-2024-12-17":           MODEL_O200K_BASE,
		"o3":                      MODEL_O200K_BASE,
		"o3-2025-01-31":           MODEL_O200K_BASE,
		"o4-mini":                 MODEL_O200K_BASE,
		"o4-mini-2025-04-16":      MODEL_O200K_BASE,
		"gpt-5":                   MODEL_O200K_BASE,
		"gpt-5-2026-01-01":        MODEL_O200K_BASE,
		"chatgpt-4o-2025-01-29":   MODEL_O200K_BASE,
		"gpt-35-turbo":            MODEL_CL100K_BASE,
		"gpt-35-turbo-16k":        MODEL_CL100K_BASE,
		"ft:gpt-4o-custom":        MODEL_O200K_BASE,
		"ft:gpt-4-custom":         MODEL_CL100K_BASE,
		"ft:gpt-3.5-turbo-custom": MODEL_CL100K_BASE,
		"ft:davinci-002-custom":   MODEL_CL100K_BASE,
		"ft:babbage-002-custom":   MODEL_CL100K_BASE,
		"gpt-oss-120b":            MODEL_O200K_HARMONY,
		"gpt-2":                   "gpt2",
	}

	for modelName, encodingName := range testCases {
		modelName := modelName
		encodingName := encodingName
		t.Run(modelName, func(t *testing.T) {
			ass := assert.New(t)
			enc, err := EncodingForModel(modelName)
			ass.NoError(err)
			if err != nil {
				return
			}
			ass.Equal(encodingName, enc.pbeEncoding.Name)
		})
	}
}

func TestO200KHarmonySpecialTokens(t *testing.T) {
	ass := assert.New(t)

	enc, err := GetEncoding(MODEL_O200K_HARMONY)
	ass.NoError(err)
	ass.Equal(MODEL_O200K_HARMONY, enc.pbeEncoding.Name)
	ass.Equal(199998, enc.pbeEncoding.SpecialTokens["<|startoftext|>"])
	ass.Equal(200012, enc.pbeEncoding.SpecialTokens["<|call|>"])
	ass.Equal(200099, enc.pbeEncoding.SpecialTokens["<|reserved_200099|>"])
	ass.Equal(201087, enc.pbeEncoding.SpecialTokens["<|reserved_201087|>"])
}

func TestDecoding(t *testing.T) {
	ass := assert.New(t)
	// enc, err := GetEncoding("cl100k_base")
	enc, err := GetEncoding(MODEL_CL100K_BASE)
	ass.Nil(err, "Encoding  init should not be nil")
	sourceTokens := []int{15339, 1917, 0, 57668, 53901, 3922, 3574, 244, 98220, 6447}
	txt := enc.Decode(sourceTokens)
	ass.Equal("hello world!你好，世界！", txt, "Decoding should be equal")
}
