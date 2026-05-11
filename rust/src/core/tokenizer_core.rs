// Copyright © Anthony DePasquale

use minijinja::Environment;
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::path::Path;
use tokenizers::{EncodeInput, Encoding, InputSequence, Model, Tokenizer};

use crate::core::error::CoreError;
use crate::core::sidecars::{self, RuntimeConfiguration, TokenizerMetadata};
use crate::core::template;
use crate::core::tokenizer_json::{self, TokenizerJsonArtifacts};

pub(crate) struct TokenizerCore {
    tokenizer: Tokenizer,
    environment: Environment<'static>,
    pub(crate) metadata: TokenizerMetadata,
}

/// Selects which addressing scheme `EncodingMetadata::offsets` uses. The
/// FFI-facing UniFFI enum (`core::uniffi_api::OffsetUnit`) translates to and
/// from this internal type.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum EncodingOffsetUnit {
    Utf8,
    UnicodeScalar,
}

#[derive(Debug, Eq, PartialEq, Serialize)]
pub(crate) struct EncodingOffsetSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct EncodingMetadata {
    pub(crate) ids: Vec<i32>,
    pub(crate) type_ids: Vec<i32>,
    pub(crate) tokens: Vec<String>,
    pub(crate) word_ids: Vec<Option<i32>>,
    pub(crate) offsets: Vec<EncodingOffsetSpan>,
    pub(crate) special_tokens_mask: Vec<i32>,
    pub(crate) attention_mask: Vec<i32>,
    pub(crate) sequence_ids: Vec<Option<i32>>,
    pub(crate) n_sequences: usize,
    pub(crate) overflowing: Vec<EncodingMetadata>,
    pub(crate) offset_unit: EncodingOffsetUnit,
}

impl EncodingMetadata {
    fn from_encoding(encoding: &Encoding, offset_unit: EncodingOffsetUnit) -> Self {
        let special_tokens_mask: Vec<i32> = encoding
            .get_special_tokens_mask()
            .iter()
            .map(|value| *value as i32)
            .collect();
        let offsets = encoding
            .get_offsets()
            .iter()
            .map(|(start, end)| EncodingOffsetSpan {
                start: *start,
                end: *end,
            })
            .collect();

        Self {
            ids: encoding.get_ids().iter().map(|id| *id as i32).collect(),
            type_ids: encoding
                .get_type_ids()
                .iter()
                .map(|id| *id as i32)
                .collect(),
            tokens: encoding.get_tokens().to_vec(),
            word_ids: encoding
                .get_word_ids()
                .iter()
                .map(|id| id.map(|id| id as i32))
                .collect(),
            offsets,
            special_tokens_mask,
            attention_mask: encoding
                .get_attention_mask()
                .iter()
                .map(|value| *value as i32)
                .collect(),
            sequence_ids: encoding
                .get_sequence_ids()
                .into_iter()
                .map(|id| id.map(|id| id as i32))
                .collect(),
            n_sequences: encoding.n_sequences(),
            overflowing: encoding
                .get_overflowing()
                .iter()
                .map(|encoding| Self::from_encoding(encoding, offset_unit))
                .collect(),
            offset_unit,
        }
    }
}

impl TokenizerCore {
    pub(crate) fn from_directory(directory: &Path) -> Result<Self, CoreError> {
        let artifacts = tokenizer_json::load_artifacts(directory)?;
        let runtime_configuration =
            sidecars::load_runtime_configuration(directory, &artifacts.metadata);
        Self::from_artifacts_and_runtime_configuration(artifacts, runtime_configuration)
    }

    pub(crate) fn from_artifacts_and_runtime_configuration(
        artifacts: TokenizerJsonArtifacts,
        runtime_configuration: RuntimeConfiguration,
    ) -> Result<Self, CoreError> {
        let metadata = Self::metadata_from_artifacts(&artifacts, runtime_configuration);
        Ok(Self {
            tokenizer: artifacts.tokenizer,
            environment: template::make_environment(),
            metadata,
        })
    }

    fn metadata_from_artifacts(
        artifacts: &TokenizerJsonArtifacts,
        runtime_configuration: RuntimeConfiguration,
    ) -> TokenizerMetadata {
        let (base_vocab_size, total_vocab_size) = Self::vocab_sizes(&artifacts.tokenizer);
        TokenizerMetadata {
            bos_token_id: resolve_token_id(
                &artifacts.tokenizer,
                runtime_configuration.bos_token.as_deref(),
            ),
            eos_token_id: resolve_token_id(
                &artifacts.tokenizer,
                runtime_configuration.eos_token.as_deref(),
            ),
            unknown_token_id: resolve_token_id(
                &artifacts.tokenizer,
                runtime_configuration.unknown_token.as_deref(),
            ),
            base_vocab_size,
            total_vocab_size,
            runtime_configuration,
        }
    }

    // Returns `(base, total)` where `base` is the model vocabulary size and
    // `total` is `base` plus added tokens not already in the model.
    //
    // Upstream `Tokenizer::get_vocab_size(true)` allocates a fresh
    // HashMap<String, u32> of the entire vocabulary just to call .len() on it,
    // which is tens of milliseconds for large vocabs. The upstream `mod.rs`
    // flags this with a `TODO ArthurZ THIS IS WRONG!` comment. We compute the
    // total directly. Added vocabularies are small in practice, so the filter
    // is cheap.
    //
    // TODO: when upstream resolves its TODO and `get_vocab_size(true)` becomes
    // O(1), drop this custom path and call the upstream API directly.
    fn vocab_sizes(tokenizer: &Tokenizer) -> (usize, usize) {
        let model = tokenizer.get_model();
        let base = model.get_vocab_size();
        let added_vocab = tokenizer.get_added_vocabulary().get_vocab();
        let unique_added = added_vocab
            .keys()
            .filter(|token| model.token_to_id(token).is_none())
            .count();
        (base, base + unique_added)
    }

    pub(crate) fn tokenize(&self, text: &str) -> Result<Vec<String>, CoreError> {
        // Use `encode` rather than `encode_fast`: upstream's `encode_fast` skips
        // populating token strings (it returns `String::with_capacity(0)` for
        // each token) as an optimization for callers that only need IDs.
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| CoreError::Internal(err.to_string()))?;
        Ok(encoding.get_tokens().to_vec())
    }

    pub(crate) fn encode(
        &self,
        text: &str,
        text_pair: Option<&str>,
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, CoreError> {
        let encoding = self
            .tokenizer
            .encode_fast(make_encode_input(text, text_pair), add_special_tokens)
            .map_err(|err| CoreError::Internal(err.to_string()))?;
        Ok(encoding.get_ids().iter().map(|id| *id as i32).collect())
    }

    pub(crate) fn encode_with_metadata(
        &self,
        text: &str,
        text_pair: Option<&str>,
        add_special_tokens: bool,
        offset_unit: EncodingOffsetUnit,
    ) -> Result<EncodingMetadata, CoreError> {
        let input = make_encode_input(text, text_pair);
        let encoding = match offset_unit {
            EncodingOffsetUnit::Utf8 => self.tokenizer.encode(input, add_special_tokens),
            EncodingOffsetUnit::UnicodeScalar => self
                .tokenizer
                .encode_char_offsets(input, add_special_tokens),
        }
        .map_err(|err| CoreError::Internal(err.to_string()))?;
        Ok(EncodingMetadata::from_encoding(&encoding, offset_unit))
    }

    pub(crate) fn encode_batch(
        &self,
        inputs: Vec<(String, Option<String>)>,
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<i32>>, CoreError> {
        let encode_inputs: Vec<EncodeInput<'_>> = inputs
            .iter()
            .map(|(text, pair)| make_encode_input(text, pair.as_deref()))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch_fast(encode_inputs, add_special_tokens)
            .map_err(|err| CoreError::Internal(err.to_string()))?;
        Ok(encodings
            .into_iter()
            .map(|encoding| encoding.get_ids().iter().map(|id| *id as i32).collect())
            .collect())
    }

    pub(crate) fn encode_batch_with_metadata(
        &self,
        inputs: Vec<(String, Option<String>)>,
        add_special_tokens: bool,
        offset_unit: EncodingOffsetUnit,
    ) -> Result<Vec<EncodingMetadata>, CoreError> {
        let encode_inputs: Vec<EncodeInput<'_>> = inputs
            .iter()
            .map(|(text, pair)| make_encode_input(text, pair.as_deref()))
            .collect();
        let encodings = match offset_unit {
            EncodingOffsetUnit::Utf8 => self
                .tokenizer
                .encode_batch(encode_inputs, add_special_tokens),
            EncodingOffsetUnit::UnicodeScalar => self
                .tokenizer
                .encode_batch_char_offsets(encode_inputs, add_special_tokens),
        }
        .map_err(|err| CoreError::Internal(err.to_string()))?;
        Ok(encodings
            .iter()
            .map(|encoding| EncodingMetadata::from_encoding(encoding, offset_unit))
            .collect())
    }

    pub(crate) fn decode(
        &self,
        token_ids: &[i32],
        skip_special_tokens: bool,
    ) -> Result<String, CoreError> {
        let ids = token_ids
            .iter()
            .map(|id| {
                u32::try_from(*id).map_err(|_| {
                    CoreError::MismatchedConfig("Token ids must be non-negative".to_owned())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.tokenizer
            .decode(&ids, skip_special_tokens)
            .map_err(|err| CoreError::Internal(err.to_string()))
    }

    pub(crate) fn convert_token_to_id(&self, token: &str) -> Option<i32> {
        self.tokenizer
            .token_to_id(token)
            .map(|token_id| token_id as i32)
    }

    pub(crate) fn convert_id_to_token(&self, token_id: i32) -> Option<String> {
        u32::try_from(token_id)
            .ok()
            .and_then(|token_id| self.tokenizer.id_to_token(token_id))
    }

    pub(crate) fn apply_chat_template(
        &self,
        template: &str,
        context: JsonValue,
        truncation: bool,
        max_length: Option<usize>,
    ) -> Result<Vec<i32>, CoreError> {
        let rendered = template::render(&self.environment, template, &context)?;
        let mut encoded = self.encode(&rendered, None, false)?;

        if truncation
            && let Some(max_length) = max_length
            && encoded.len() > max_length
        {
            encoded.truncate(max_length);
        }

        Ok(encoded)
    }
}

fn resolve_token_id(tokenizer: &Tokenizer, token: Option<&str>) -> Option<i32> {
    token
        .and_then(|token| tokenizer.token_to_id(token))
        .map(|token_id| token_id as i32)
}

fn make_encode_input<'a>(text: &'a str, text_pair: Option<&'a str>) -> EncodeInput<'a> {
    match text_pair {
        Some(pair) => EncodeInput::Dual(
            InputSequence::Raw(text.into()),
            InputSequence::Raw(pair.into()),
        ),
        None => EncodeInput::Single(InputSequence::Raw(text.into())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn offline_fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Tests/TokenizersTests/Resources")
    }

    #[test]
    fn creates_tokenizer_core_from_offline_fixture_directory() {
        let directory = offline_fixture_directory();
        let core = TokenizerCore::from_directory(&directory)
            .unwrap_or_else(|error| panic!("failed to build tokenizer core: {error}"));

        assert_eq!(core.convert_token_to_id("<unk>"), Some(3));
        assert_eq!(
            core.metadata.runtime_configuration.unknown_token.as_deref(),
            Some("<unk>")
        );
    }

    #[test]
    fn encodes_metadata_with_unicode_scalar_offsets() {
        let directory = offline_fixture_directory();
        let core = TokenizerCore::from_directory(&directory)
            .unwrap_or_else(|error| panic!("failed to build tokenizer core: {error}"));

        let encoding = core
            .encode_with_metadata("hello world", None, true, EncodingOffsetUnit::UnicodeScalar)
            .unwrap_or_else(|error| panic!("failed to encode metadata: {error}"));

        assert_eq!(encoding.ids.len(), encoding.tokens.len());
        assert_eq!(encoding.ids.len(), encoding.offsets.len());
        assert!(
            encoding
                .offsets
                .iter()
                .any(|offset| offset.end > offset.start)
        );
        assert_eq!(encoding.offset_unit, EncodingOffsetUnit::UnicodeScalar);
        for (offset, special_token) in encoding.offsets.iter().zip(&encoding.special_tokens_mask) {
            if *special_token != 0 {
                assert_eq!(offset, &EncodingOffsetSpan { start: 0, end: 0 });
            }
        }

        let payload = serde_json::to_value(&encoding)
            .unwrap_or_else(|error| panic!("failed to serialize encoding metadata: {error}"));
        assert_eq!(payload["offset_unit"], "unicodeScalar");
        assert_eq!(payload["n_sequences"], 1);
        assert!(payload.get("type_ids").is_some());
        assert!(payload.get("special_tokens_mask").is_some());
    }

    #[test]
    fn metadata_offset_unit_controls_offset_coordinates() {
        let directory = offline_fixture_directory();
        let core = TokenizerCore::from_directory(&directory)
            .unwrap_or_else(|error| panic!("failed to build tokenizer core: {error}"));

        let text = "hé";
        let utf8 = core
            .encode_with_metadata(text, None, false, EncodingOffsetUnit::Utf8)
            .unwrap_or_else(|error| panic!("failed to encode utf8 metadata: {error}"));
        let unicode_scalar = core
            .encode_with_metadata(text, None, false, EncodingOffsetUnit::UnicodeScalar)
            .unwrap_or_else(|error| panic!("failed to encode unicode-scalar metadata: {error}"));

        assert_eq!(utf8.offset_unit, EncodingOffsetUnit::Utf8);
        assert_eq!(
            unicode_scalar.offset_unit,
            EncodingOffsetUnit::UnicodeScalar
        );
        assert_ne!(utf8.offsets, unicode_scalar.offsets);

        let utf8_end = utf8.offsets.iter().map(|offset| offset.end).max();
        let unicode_scalar_end = unicode_scalar.offsets.iter().map(|offset| offset.end).max();

        assert_eq!(utf8_end, Some(text.len()));
        assert_eq!(unicode_scalar_end, Some(text.chars().count()));
    }
}
