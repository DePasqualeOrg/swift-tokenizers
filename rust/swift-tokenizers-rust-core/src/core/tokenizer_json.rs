use serde_json::Value as JsonValue;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::core::error::CoreError;

#[derive(Debug, Clone)]
pub(crate) struct TokenizerJsonMetadata {
    pub(crate) bos_token: Option<String>,
    pub(crate) eos_token: Option<String>,
    pub(crate) unknown_token: Option<String>,
    pub(crate) fuse_unknown_tokens: bool,
    pub(crate) post_processor_type: Option<String>,
}

pub(crate) struct TokenizerJsonArtifacts {
    pub(crate) tokenizer: Tokenizer,
    pub(crate) metadata: TokenizerJsonMetadata,
}

pub(crate) fn load_artifacts(directory: &Path) -> Result<TokenizerJsonArtifacts, CoreError> {
    let tokenizer_bytes = read_required_bytes(&directory.join("tokenizer.json"))?;
    let mut tokenizer_data: Option<JsonValue> = None;
    let tokenizer = load_tokenizer(&tokenizer_bytes, &mut tokenizer_data)?;
    let tokenizer_data = match tokenizer_data {
        Some(tokenizer_data) => tokenizer_data,
        None => serde_json::from_slice(&tokenizer_bytes)
            .map_err(|err| CoreError::MismatchedConfig(err.to_string()))?,
    };
    let metadata = extract_metadata_from_value(&tokenizer_data);

    Ok(TokenizerJsonArtifacts { tokenizer, metadata })
}

fn read_required_bytes(path: &Path) -> Result<Vec<u8>, CoreError> {
    fs::read(path).map_err(|_| CoreError::MissingConfig)
}

fn load_tokenizer(bytes: &[u8], tokenizer_data: &mut Option<JsonValue>) -> Result<Tokenizer, CoreError> {
    if let Ok(tokenizer) = Tokenizer::from_bytes(bytes) {
        return Ok(tokenizer);
    }

    let tokenizer_data = tokenizer_data.get_or_insert_with(|| {
        serde_json::from_slice(bytes).expect("fallback tokenizer JSON parse should match earlier load failure path")
    });
    let tokenizer_bytes = prepare_tokenizer_bytes(tokenizer_data)?;
    Tokenizer::from_bytes(&tokenizer_bytes)
        .map_err(|err| CoreError::MismatchedConfig(err.to_string()))
}

fn normalize_added_tokens(tokenizer_data: &mut JsonValue) {
    let Some(added_tokens) = tokenizer_data
        .get_mut("added_tokens")
        .and_then(JsonValue::as_array_mut)
    else {
        return;
    };

    for added_token in added_tokens {
        let Some(added_token) = added_token.as_object_mut() else {
            continue;
        };

        added_token
            .entry("single_word".to_owned())
            .or_insert(JsonValue::Bool(false));
        added_token
            .entry("lstrip".to_owned())
            .or_insert(JsonValue::Bool(false));
        added_token
            .entry("rstrip".to_owned())
            .or_insert(JsonValue::Bool(false));
        added_token
            .entry("normalized".to_owned())
            .or_insert(JsonValue::Bool(false));
        added_token
            .entry("special".to_owned())
            .or_insert(JsonValue::Bool(false));
    }
}

fn prepare_tokenizer_bytes(tokenizer_data: &mut JsonValue) -> Result<Vec<u8>, CoreError> {
    normalize_added_tokens(tokenizer_data);
    serde_json::to_vec(tokenizer_data).map_err(|err| CoreError::Internal(err.to_string()))
}

pub(crate) fn extract_token_string_array(value: Option<&JsonValue>) -> Vec<String> {
    let Some(JsonValue::Array(values)) = value else {
        return Vec::new();
    };

    values
        .iter()
        .filter_map(|value| extract_token_string(Some(value)))
        .collect()
}

pub(crate) fn extract_token_string(value: Option<&JsonValue>) -> Option<String> {
    match value {
        Some(JsonValue::String(value)) => Some(value.clone()),
        Some(JsonValue::Object(map)) => map
            .get("content")
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned),
        _ => None,
    }
}

fn extract_metadata_from_value(tokenizer_data: &JsonValue) -> TokenizerJsonMetadata {
    let model = tokenizer_data.get("model");
    let unknown_token = model
        .and_then(|model| model.get("unk_token"))
        .and_then(JsonValue::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| extract_token_string(tokenizer_data.get("unk_token")))
        .or_else(|| {
            let unk_id = model?.get("unk_id").and_then(JsonValue::as_u64)? as usize;
            let vocab = model?.get("vocab")?.as_array()?;
            let token = vocab.get(unk_id)?;
            if let Some(value) = token.as_array() {
                return value.first().and_then(JsonValue::as_str).map(ToOwned::to_owned);
            }
            token.get("token")
                .and_then(JsonValue::as_str)
                .map(ToOwned::to_owned)
        });
    let fuse_unknown_tokens = model
        .and_then(|model| model.get("fuse_unk"))
        .and_then(JsonValue::as_bool)
        .unwrap_or_else(|| {
            model
                .and_then(|model| model.get("type"))
                .and_then(JsonValue::as_str)
                == Some("Unigram")
        });

    TokenizerJsonMetadata {
        bos_token: extract_token_string(tokenizer_data.get("bos_token")),
        eos_token: extract_token_string(tokenizer_data.get("eos_token")),
        unknown_token,
        fuse_unknown_tokens,
        post_processor_type: tokenizer_data
            .get("post_processor")
            .and_then(|post_processor| post_processor.get("type"))
            .and_then(JsonValue::as_str)
            .map(ToOwned::to_owned),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn offline_fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../Tests/TokenizersTests/Resources")
    }

    #[test]
    fn loads_offline_fixture_tokenizer() {
        let directory = offline_fixture_directory();
        let artifacts = load_artifacts(&directory)
            .unwrap_or_else(|error| panic!("failed to load tokenizer fixture: {error}"));

        assert_eq!(artifacts.metadata.unknown_token.as_deref(), Some("<unk>"));
        assert_eq!(artifacts.tokenizer.token_to_id("<unk>"), Some(3));
    }
}
