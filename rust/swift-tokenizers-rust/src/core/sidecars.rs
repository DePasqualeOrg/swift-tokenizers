use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::fs;
use std::path::Path;

use crate::core::tokenizer_json::{self, TokenizerJsonMetadata};

// `tokenizer.json`'s `post_processor` is the sole source of truth for which
// special tokens get prepended or appended. Python `transformers` v5 takes
// the same stance: `PreTrainedTokenizerBase.from_pretrained` pops
// `add_bos_token` / `add_eos_token` from `init_kwargs` whenever a
// `tokenizer.json` is present, so they never reach
// `PreTrainedTokenizerFast.update_post_processor`. A tokenizer that needs a
// leading bos must carry it in its `tokenizer.json` post-processor.
//
// `fuse_unk` is intentionally absent: upstream serializes it on each model
// block in `tokenizer.json` (BPE / Unigram), and the Rust `tokenizers` library
// applies the fusion inside its tokenize path. Neither Python `transformers` v5
// nor the `tokenizers` library ever reads the flag from `tokenizer_config.json`,
// so mirroring it into the runtime config would be dead duplication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RuntimeConfiguration {
    pub(crate) bos_token: Option<String>,
    pub(crate) eos_token: Option<String>,
    pub(crate) unknown_token: Option<String>,
    pub(crate) sep_token: Option<String>,
    pub(crate) pad_token: Option<String>,
    pub(crate) cls_token: Option<String>,
    pub(crate) mask_token: Option<String>,
    pub(crate) additional_special_tokens: Vec<String>,
    pub(crate) clean_up_tokenization_spaces: bool,
    pub(crate) model_max_length: Option<u64>,
    pub(crate) chat_template: Option<JsonValue>,
}

#[derive(Debug, Serialize)]
pub(crate) struct TokenizerMetadata {
    pub(crate) runtime_configuration: RuntimeConfiguration,
    pub(crate) bos_token_id: Option<i32>,
    pub(crate) eos_token_id: Option<i32>,
    pub(crate) unknown_token_id: Option<i32>,
}

pub(crate) fn load_runtime_configuration(
    directory: &Path,
    tokenizer_metadata: &TokenizerJsonMetadata,
) -> RuntimeConfiguration {
    load_runtime_configuration_impl(directory, Some(tokenizer_metadata))
}

pub(crate) fn load_runtime_configuration_only(directory: &Path) -> RuntimeConfiguration {
    load_runtime_configuration_impl(directory, None)
}

fn load_runtime_configuration_impl(
    directory: &Path,
    tokenizer_metadata: Option<&TokenizerJsonMetadata>,
) -> RuntimeConfiguration {
    let tokenizer_config = read_optional_json_file(&directory.join("tokenizer_config.json"));
    let chat_template_override = load_chat_template_override(directory);

    let bos_token = tokenizer_json::extract_token_string(
        tokenizer_config
            .as_ref()
            .and_then(|config| config.get("bos_token")),
    )
    .or_else(|| tokenizer_metadata.and_then(|metadata| metadata.bos_token.clone()));
    let eos_token = tokenizer_json::extract_token_string(
        tokenizer_config
            .as_ref()
            .and_then(|config| config.get("eos_token")),
    )
    .or_else(|| tokenizer_metadata.and_then(|metadata| metadata.eos_token.clone()));
    let unknown_token = tokenizer_json::extract_token_string(
        tokenizer_config
            .as_ref()
            .and_then(|config| config.get("unk_token")),
    )
    .or_else(|| tokenizer_metadata.and_then(|metadata| metadata.unknown_token.clone()));

    RuntimeConfiguration {
        bos_token,
        eos_token,
        unknown_token,
        sep_token: tokenizer_json::extract_token_string(
            tokenizer_config
                .as_ref()
                .and_then(|config| config.get("sep_token")),
        ),
        pad_token: tokenizer_json::extract_token_string(
            tokenizer_config
                .as_ref()
                .and_then(|config| config.get("pad_token")),
        ),
        cls_token: tokenizer_json::extract_token_string(
            tokenizer_config
                .as_ref()
                .and_then(|config| config.get("cls_token")),
        ),
        mask_token: tokenizer_json::extract_token_string(
            tokenizer_config
                .as_ref()
                .and_then(|config| config.get("mask_token")),
        ),
        additional_special_tokens: tokenizer_json::extract_token_string_array(
            tokenizer_config
                .as_ref()
                .and_then(|config| config.get("additional_special_tokens")),
        ),
        clean_up_tokenization_spaces: tokenizer_config
            .as_ref()
            .and_then(|config| config.get("clean_up_tokenization_spaces"))
            .and_then(JsonValue::as_bool)
            .unwrap_or(true),
        model_max_length: tokenizer_config
            .as_ref()
            .and_then(|config| config.get("model_max_length"))
            .and_then(extract_u64),
        chat_template: chat_template_override.or_else(|| {
            tokenizer_config
                .as_ref()
                .and_then(|config| config.get("chat_template"))
                .filter(|value| !value.is_null())
                .cloned()
        }),
    }
}

fn extract_u64(value: &JsonValue) -> Option<u64> {
    value
        .as_u64()
        .or_else(|| value.as_i64().and_then(|value| u64::try_from(value).ok()))
}

fn read_optional_json_file(path: &Path) -> Option<JsonValue> {
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

fn load_chat_template_override(directory: &Path) -> Option<JsonValue> {
    let chat_template_jinja_path = directory.join("chat_template.jinja");
    if let Ok(template) = fs::read_to_string(&chat_template_jinja_path) {
        return Some(JsonValue::String(template));
    }

    let chat_template_json = read_optional_json_file(&directory.join("chat_template.json"))?;
    let template = chat_template_json.get("chat_template")?;
    if template.is_null() {
        return None;
    }
    Some(template.clone())
}
