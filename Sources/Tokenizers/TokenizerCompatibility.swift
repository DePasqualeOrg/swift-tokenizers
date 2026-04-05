import Foundation

package enum TokenizerCompatibility {
    // `tokenizer_class` / `model_type` come from Hugging Face transformers sidecars, not from the
    // lower-level tokenizer engine itself. Some names map to wrapper-level behavior rather than a
    // distinct core algorithm. For example, Python transformers' `NllbTokenizer` is a BPE-based
    // wrapper with language-code handling in `models/nllb/tokenization_nllb.py`.
    package static let modelTypeToTokenizerClass: [String: String] = [
        "bert": "BertTokenizer",
        "code_llama": "CodeLlamaTokenizer",
        "codegen": "GPT2Tokenizer",
        "cohere": "CohereTokenizer",
        "distilbert": "BertTokenizer",
        "gemma": "GemmaTokenizer",
        "gemma2": "GemmaTokenizer",
        "gpt2": "GPT2Tokenizer",
        "llama": "LlamaTokenizer",
        "qwen2": "Qwen2Tokenizer",
        "roberta": "RobertaTokenizer",
        "t5": "T5Tokenizer",
        "whisper": "WhisperTokenizer",
        "xlm-roberta": "XLMRobertaTokenizer",
    ]

    // Rust mode does not mirror the Swift registry one-for-one. This list tracks the
    // `tokenizer_class` names we accept in strict mode before delegating to upstream
    // `huggingface/tokenizers` plus targeted compatibility shims.
    private static let rustSupportedTokenizerNames: Set<String> = [
        "BertTokenizer",
        "CodeGenTokenizer",
        "CodeLlamaTokenizer",
        "CohereTokenizer",
        "DistilbertTokenizer",
        "DistilBertTokenizer",
        "FalconTokenizer",
        "GemmaTokenizer",
        "GPT2Tokenizer",
        "GPTNeoXTokenizer",
        "InternLM2Tokenizer",
        "LlamaTokenizer",
        "PreTrainedTokenizer",
        "Qwen2Tokenizer",
        "Qwen3Tokenizer",
        "RobertaTokenizer",
        "T5Tokenizer",
        "TokenizersBackend",
        "WhisperTokenizer",
        "XLMRobertaTokenizer",
        "Xlm-RobertaTokenizer",
    ]

    package static func resolvedTokenizerClass(tokenizerClass: String?, modelType: String?) -> String? {
        tokenizerClass ?? modelType.flatMap { modelTypeToTokenizerClass[$0] }
    }

    package static func normalizedTokenizerName(_ name: String) -> String {
        name.replacingOccurrences(of: "Fast", with: "")
    }

    package static func resolvedTokenizerName(tokenizerClass: String?, modelType: String?) -> String? {
        resolvedTokenizerClass(tokenizerClass: tokenizerClass, modelType: modelType).map(normalizedTokenizerName)
    }

    package static func validateResolvedTokenizerName(
        tokenizerClass: String?,
        modelType: String?,
        strict: Bool,
        isSupported: (String) -> Bool
    ) throws -> String {
        guard let tokenizerName = resolvedTokenizerName(tokenizerClass: tokenizerClass, modelType: modelType) else {
            throw TokenizerError.missingTokenizerClassInConfig
        }

        if !isSupported(tokenizerName), strict {
            throw TokenizerError.unsupportedTokenizer(tokenizerName)
        }

        return tokenizerName
    }

    package static func isRustSupportedTokenizer(named tokenizerName: String) -> Bool {
        rustSupportedTokenizerNames.contains(normalizedTokenizerName(tokenizerName))
    }
}
