//! UniFFI-exported surface for the Swift backend. This is the sole FFI surface
//! between the Rust core and the Swift adapter in `RustTokenizer.swift`.

use std::path::Path;
use std::sync::Arc;

use serde_json::Value as JsonValue;

use crate::core::error::CoreError;
use crate::core::sidecars::{self, RuntimeConfiguration as CoreRuntimeConfiguration};
use crate::core::template;
use crate::core::tokenizer_core::{
    EncodingMetadata as CoreEncodingMetadata, EncodingOffsetSpan as CoreEncodingOffsetSpan,
    EncodingOffsetUnit, TokenizerCore,
};
use crate::core::tokenizer_json;

/// Errors crossing the FFI boundary. The variant set mirrors `core::error::CoreError`
/// minus the FFI-internal "input or output pointer was null" cases that no longer
/// apply once UniFFI manages marshalling. The Swift adapter remaps these onto the
/// existing public `TokenizerError` enum.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum TokenizerError {
    #[error("Tokenizer configuration is missing.")]
    MissingConfig,
    #[error("Chat template error: {message}")]
    ChatTemplate { message: String },
    #[error("Tokenizer configuration mismatch: {message}")]
    MismatchedConfig { message: String },
    #[error("{message}")]
    Internal { message: String },
}

impl From<CoreError> for TokenizerError {
    fn from(error: CoreError) -> Self {
        match error {
            CoreError::MissingConfig => Self::MissingConfig,
            CoreError::ChatTemplate(message) => Self::ChatTemplate { message },
            CoreError::MismatchedConfig(message) => Self::MismatchedConfig { message },
            CoreError::Internal(message) => Self::Internal { message },
        }
    }
}

/// Discriminant byte selecting which addressing scheme `EncodingMetadata::offset_spans`
/// uses. Mirrors the public Swift `OffsetUnit` and the internal
/// `core::tokenizer_core::EncodingOffsetUnit`.
#[derive(Debug, Clone, Copy, Eq, PartialEq, uniffi::Enum)]
pub enum OffsetUnit {
    Utf8,
    UnicodeScalar,
}

impl From<OffsetUnit> for EncodingOffsetUnit {
    fn from(value: OffsetUnit) -> Self {
        match value {
            OffsetUnit::Utf8 => Self::Utf8,
            OffsetUnit::UnicodeScalar => Self::UnicodeScalar,
        }
    }
}

impl From<EncodingOffsetUnit> for OffsetUnit {
    fn from(value: EncodingOffsetUnit) -> Self {
        match value {
            EncodingOffsetUnit::Utf8 => Self::Utf8,
            EncodingOffsetUnit::UnicodeScalar => Self::UnicodeScalar,
        }
    }
}

/// Half-open offset span associated with a token, expressed in the encoding's
/// `OffsetUnit`.
#[derive(Debug, Clone, uniffi::Record)]
pub struct OffsetSpan {
    pub start: u64,
    pub end: u64,
}

impl From<&CoreEncodingOffsetSpan> for OffsetSpan {
    fn from(span: &CoreEncodingOffsetSpan) -> Self {
        Self {
            start: span.start as u64,
            end: span.end as u64,
        }
    }
}

/// Per-encoding payload carried by both the primary encoding and its overflow
/// children. See `Encoding` for the recursion-flattening rationale.
#[derive(Debug, uniffi::Record)]
pub struct EncodingMetadata {
    pub token_ids: Vec<i32>,
    pub token_type_ids: Vec<i32>,
    pub tokens: Vec<String>,
    pub word_indices: Vec<Option<i32>>,
    pub offset_spans: Vec<OffsetSpan>,
    pub special_tokens_mask: Vec<i32>,
    pub attention_mask: Vec<i32>,
    pub sequence_indices: Vec<Option<i32>>,
    pub sequence_count: u64,
    pub offset_unit: OffsetUnit,
}

impl From<CoreEncodingMetadata> for EncodingMetadata {
    fn from(metadata: CoreEncodingMetadata) -> Self {
        // The `Encoding` Record is two-level flat (primary + Vec<children>);
        // it has nowhere to put grandchildren. The upstream `tokenizers` crate
        // never produces them today (its truncator yields children with empty
        // `overflowing`), but this assert pins the invariant so a future
        // upstream change becomes a loud test failure instead of silent data
        // loss. See the `Encoding` doc comment for the migration path.
        debug_assert!(
            metadata.overflowing.is_empty(),
            "EncodingMetadata::from is dropping a non-empty `overflowing` field; \
             upstream `tokenizers` may now produce nested overflow chains. \
             Switch the FFI to Option A (opaque Encoding object) per the \
             comment on `Encoding` in this file.",
        );
        Self {
            token_ids: metadata.ids,
            token_type_ids: metadata.type_ids,
            tokens: metadata.tokens,
            word_indices: metadata.word_ids,
            offset_spans: metadata.offsets.iter().map(OffsetSpan::from).collect(),
            special_tokens_mask: metadata.special_tokens_mask,
            attention_mask: metadata.attention_mask,
            sequence_indices: metadata.sequence_ids,
            sequence_count: metadata.n_sequences as u64,
            offset_unit: metadata.offset_unit.into(),
        }
    }
}

/// Encoding modeling: Option B (flat DTO) per `docs/uniffi-migration.md`
/// § "Encoding modeling". The upstream `tokenizers` crate populates
/// `Encoding::overflowing` only via the truncator, which always produces
/// children whose own `overflowing` field is empty — so the recursive shape in
/// the public Swift `TokenizerEncoding` carries at most one level of nesting,
/// and flattening is lossless. UniFFI Records cannot recursively contain
/// themselves, which is why the ABI uses this two-level form. If a future
/// upstream change ever populates a child's overflow, the Swift adapter will
/// silently lose those grandchildren — switch to Option A (opaque object) at
/// that point.
#[derive(Debug, uniffi::Record)]
pub struct Encoding {
    pub primary: EncodingMetadata,
    pub overflowing: Vec<EncodingMetadata>,
}

impl From<CoreEncodingMetadata> for Encoding {
    fn from(mut metadata: CoreEncodingMetadata) -> Self {
        // Take the parent's `overflowing` first, then map the children. Once
        // `metadata.overflowing` is empty, the `debug_assert!` inside
        // `EncodingMetadata::from` only fires for the *child* encodings handled
        // here — exactly the case we want to catch (a grandchild we'd silently
        // drop). The parent's own pass-through is no-op for the assert.
        let overflowing = std::mem::take(&mut metadata.overflowing)
            .into_iter()
            .map(EncodingMetadata::from)
            .collect();
        Self {
            primary: EncodingMetadata::from(metadata),
            overflowing,
        }
    }
}

/// Pair of input texts for a single batch entry. Mirrors the
/// `tokenizers::EncodeInput::Single` / `EncodeInput::Dual` shape.
#[derive(Debug, uniffi::Record)]
pub struct EncodeInput {
    pub text: String,
    pub text_pair: Option<String>,
}

#[derive(Debug, Clone, uniffi::Record)]
pub struct NamedChatTemplate {
    pub name: String,
    pub template: String,
}

/// How chat-template content is sourced. Mirrors the public Swift
/// `TokenizerRuntimeConfiguration.ChatTemplateSource` shape verbatim.
#[derive(Debug, Clone, uniffi::Enum)]
pub enum ChatTemplateSource {
    None,
    Literal { template: String },
    Named { templates: Vec<NamedChatTemplate> },
}

/// Special tokens, max length, and chat-template settings sourced from the
/// tokenizer's `tokenizer_config.json` and chat-template sidecars.
#[derive(Debug, Clone, uniffi::Record)]
pub struct RuntimeConfiguration {
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unknown_token: Option<String>,
    pub sep_token: Option<String>,
    pub pad_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Vec<String>,
    pub clean_up_tokenization_spaces: bool,
    pub model_max_length: Option<u64>,
    pub chat_template: ChatTemplateSource,
}

impl TryFrom<CoreRuntimeConfiguration> for RuntimeConfiguration {
    type Error = TokenizerError;

    fn try_from(value: CoreRuntimeConfiguration) -> Result<Self, Self::Error> {
        let chat_template = chat_template_source_from_json(value.chat_template)?;
        Ok(Self {
            bos_token: value.bos_token,
            eos_token: value.eos_token,
            unknown_token: value.unknown_token,
            sep_token: value.sep_token,
            pad_token: value.pad_token,
            cls_token: value.cls_token,
            mask_token: value.mask_token,
            additional_special_tokens: value.additional_special_tokens,
            clean_up_tokenization_spaces: value.clean_up_tokenization_spaces,
            model_max_length: value.model_max_length,
            chat_template,
        })
    }
}

impl From<RuntimeConfiguration> for CoreRuntimeConfiguration {
    fn from(value: RuntimeConfiguration) -> Self {
        Self {
            bos_token: value.bos_token,
            eos_token: value.eos_token,
            unknown_token: value.unknown_token,
            sep_token: value.sep_token,
            pad_token: value.pad_token,
            cls_token: value.cls_token,
            mask_token: value.mask_token,
            additional_special_tokens: value.additional_special_tokens,
            clean_up_tokenization_spaces: value.clean_up_tokenization_spaces,
            model_max_length: value.model_max_length,
            chat_template: chat_template_source_to_json(value.chat_template),
        }
    }
}

fn chat_template_source_from_json(
    value: Option<JsonValue>,
) -> Result<ChatTemplateSource, TokenizerError> {
    match value {
        None | Some(JsonValue::Null) => Ok(ChatTemplateSource::None),
        Some(JsonValue::String(template)) => Ok(ChatTemplateSource::Literal { template }),
        Some(JsonValue::Array(entries)) => {
            let mut templates = Vec::with_capacity(entries.len());
            for entry in entries {
                let name = entry
                    .get("name")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| TokenizerError::MismatchedConfig {
                        message: "Named chat template entry is missing a string `name` field"
                            .to_owned(),
                    })?
                    .to_owned();
                let template = entry
                    .get("template")
                    .and_then(JsonValue::as_str)
                    .ok_or_else(|| TokenizerError::MismatchedConfig {
                        message: "Named chat template entry is missing a string `template` field"
                            .to_owned(),
                    })?
                    .to_owned();
                templates.push(NamedChatTemplate { name, template });
            }
            Ok(ChatTemplateSource::Named { templates })
        }
        Some(other) => Err(TokenizerError::MismatchedConfig {
            message: format!(
                "Unsupported chat_template JSON shape: expected null, string, or array, got {}",
                shape_name(&other)
            ),
        }),
    }
}

fn chat_template_source_to_json(source: ChatTemplateSource) -> Option<JsonValue> {
    match source {
        ChatTemplateSource::None => None,
        ChatTemplateSource::Literal { template } => Some(JsonValue::String(template)),
        ChatTemplateSource::Named { templates } => Some(JsonValue::Array(
            templates
                .into_iter()
                .map(|template| {
                    let mut object = serde_json::Map::with_capacity(2);
                    object.insert("name".to_owned(), JsonValue::String(template.name));
                    object.insert("template".to_owned(), JsonValue::String(template.template));
                    JsonValue::Object(object)
                })
                .collect(),
        )),
    }
}

fn shape_name(value: &JsonValue) -> &'static str {
    match value {
        JsonValue::Null => "null",
        JsonValue::Bool(_) => "boolean",
        JsonValue::Number(_) => "number",
        JsonValue::String(_) => "string",
        JsonValue::Array(_) => "array",
        JsonValue::Object(_) => "object",
    }
}

/// Read-once metadata returned alongside the tokenizer handle from each
/// constructor. Mirrors `st_tokenizer_descriptor_t` minus the lifetime
/// boilerplate that UniFFI handles.
#[derive(Debug, uniffi::Record)]
pub struct TokenizerDescriptor {
    pub runtime_configuration: RuntimeConfiguration,
    pub bos_token_id: Option<i32>,
    pub eos_token_id: Option<i32>,
    pub unknown_token_id: Option<i32>,
    pub base_vocab_size: u64,
    pub total_vocab_size: u64,
}

#[derive(uniffi::Object)]
pub struct Tokenizer {
    core: TokenizerCore,
}

#[uniffi::export]
impl Tokenizer {
    /// Builds a tokenizer from a directory containing `tokenizer.json` plus
    /// the standard sidecar files (`tokenizer_config.json`, optional
    /// `chat_template.{jinja,json}`).
    #[uniffi::constructor]
    pub fn from_directory(directory_path: String) -> Result<Arc<Self>, TokenizerError> {
        let core = TokenizerCore::from_directory(Path::new(&directory_path))?;
        Ok(Arc::new(Self { core }))
    }

    /// Builds a tokenizer from a directory using a runtime configuration
    /// supplied by the caller. The directory must still contain `tokenizer.json`,
    /// but sidecar lookups are skipped — every special token, max length, and
    /// chat template comes from `runtime_configuration`.
    #[uniffi::constructor]
    pub fn from_directory_with_runtime_configuration(
        directory_path: String,
        runtime_configuration: RuntimeConfiguration,
    ) -> Result<Arc<Self>, TokenizerError> {
        let artifacts = tokenizer_json::load_artifacts(Path::new(&directory_path))?;
        let core = TokenizerCore::from_artifacts_and_runtime_configuration(
            artifacts,
            runtime_configuration.into(),
        )?;
        Ok(Arc::new(Self { core }))
    }

    /// Returns the read-once metadata captured at construction time.
    pub fn descriptor(&self) -> Result<TokenizerDescriptor, TokenizerError> {
        let metadata = self.core.metadata.clone();
        Ok(TokenizerDescriptor {
            runtime_configuration: metadata.runtime_configuration.try_into()?,
            bos_token_id: metadata.bos_token_id,
            eos_token_id: metadata.eos_token_id,
            unknown_token_id: metadata.unknown_token_id,
            base_vocab_size: metadata.base_vocab_size as u64,
            total_vocab_size: metadata.total_vocab_size as u64,
        })
    }

    pub fn tokenize(&self, text: String) -> Result<Vec<String>, TokenizerError> {
        Ok(self.core.tokenize(&text)?)
    }

    pub fn encode(
        &self,
        text: String,
        text_pair: Option<String>,
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, TokenizerError> {
        Ok(self
            .core
            .encode(&text, text_pair.as_deref(), add_special_tokens)?)
    }

    pub fn encode_with_metadata(
        &self,
        text: String,
        text_pair: Option<String>,
        add_special_tokens: bool,
        offset_unit: OffsetUnit,
    ) -> Result<Encoding, TokenizerError> {
        let metadata = self.core.encode_with_metadata(
            &text,
            text_pair.as_deref(),
            add_special_tokens,
            offset_unit.into(),
        )?;
        Ok(metadata.into())
    }

    pub fn encode_batch(
        &self,
        inputs: Vec<EncodeInput>,
        add_special_tokens: bool,
    ) -> Result<Vec<Vec<i32>>, TokenizerError> {
        let parsed_inputs = inputs
            .into_iter()
            .map(|input| (input.text, input.text_pair))
            .collect();
        Ok(self.core.encode_batch(parsed_inputs, add_special_tokens)?)
    }

    pub fn encode_batch_with_metadata(
        &self,
        inputs: Vec<EncodeInput>,
        add_special_tokens: bool,
        offset_unit: OffsetUnit,
    ) -> Result<Vec<Encoding>, TokenizerError> {
        let parsed_inputs = inputs
            .into_iter()
            .map(|input| (input.text, input.text_pair))
            .collect();
        let encodings = self.core.encode_batch_with_metadata(
            parsed_inputs,
            add_special_tokens,
            offset_unit.into(),
        )?;
        Ok(encodings.into_iter().map(Encoding::from).collect())
    }

    pub fn decode(
        &self,
        token_ids: Vec<i32>,
        skip_special_tokens: bool,
    ) -> Result<String, TokenizerError> {
        Ok(self.core.decode(&token_ids, skip_special_tokens)?)
    }

    pub fn convert_token_to_id(&self, token: String) -> Option<i32> {
        self.core.convert_token_to_id(&token)
    }

    pub fn convert_id_to_token(&self, token_id: i32) -> Option<String> {
        self.core.convert_id_to_token(token_id)
    }

    /// Renders a chat template to token IDs. `context_json` is the JSON
    /// representation of the template variables (`messages`, `tools`, etc.);
    /// the Swift adapter stringifies its `[String: Any]` context once before
    /// calling here.
    pub fn apply_chat_template(
        &self,
        template: String,
        context_json: String,
        truncation: bool,
        max_length: Option<u64>,
    ) -> Result<Vec<i32>, TokenizerError> {
        let context: JsonValue =
            serde_json::from_str(&context_json).map_err(|err| TokenizerError::Internal {
                message: format!("invalid context JSON: {err}"),
            })?;
        let max_length = max_length
            .map(|value| {
                usize::try_from(value).map_err(|_| TokenizerError::Internal {
                    message: "max_length does not fit in usize".to_owned(),
                })
            })
            .transpose()?;
        Ok(self
            .core
            .apply_chat_template(&template, context, truncation, max_length)?)
    }
}

/// Reads the tokenizer's runtime configuration from a directory without
/// constructing a tokenizer. Mirrors `st_load_tokenizer_runtime_configuration`.
#[uniffi::export]
pub fn load_runtime_configuration(
    directory_path: String,
) -> Result<RuntimeConfiguration, TokenizerError> {
    let configuration = sidecars::load_runtime_configuration_only(Path::new(&directory_path));
    configuration.try_into()
}

/// Renders a Jinja chat template to a string. `context_json` is the JSON
/// representation of the template variables. Mirrors `st_render_template`.
#[uniffi::export]
pub fn render_template(template: String, context_json: String) -> Result<String, TokenizerError> {
    let context: JsonValue =
        serde_json::from_str(&context_json).map_err(|err| TokenizerError::Internal {
            message: format!("invalid context JSON: {err}"),
        })?;
    let environment = template::make_environment();
    Ok(template::render(&environment, &template, &context)?)
}
