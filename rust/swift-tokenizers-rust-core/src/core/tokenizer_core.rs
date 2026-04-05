use minijinja::Environment;
use serde_json::Value as JsonValue;
use std::path::Path;
use tokenizers::Tokenizer;

use crate::core::error::CoreError;
use crate::core::sidecars::{self, RuntimeConfiguration, TokenizerMetadata};
use crate::core::template;
use crate::core::tokenizer_json::{self, TokenizerJsonArtifacts};

pub(crate) struct TokenizerCore {
    tokenizer: Tokenizer,
    environment: Environment<'static>,
    pub(crate) metadata: TokenizerMetadata,
    llama_post_processor_compatibility: Option<LlamaPostProcessorCompatibility>,
}

struct LlamaPostProcessorCompatibility {
    bos_token_id: Option<i32>,
    eos_token_id: Option<i32>,
    add_bos_token: bool,
    add_eos_token: bool,
}

impl TokenizerCore {
    pub(crate) fn from_directory(directory: &Path) -> Result<Self, CoreError> {
        let artifacts = tokenizer_json::load_artifacts(directory)?;
        let runtime_configuration =
            sidecars::load_runtime_configuration(directory, &artifacts.metadata);
        Self::from_artifacts_and_runtime_configuration(
            artifacts,
            runtime_configuration,
        )
    }

    pub(crate) fn from_artifacts_and_runtime_configuration(
        artifacts: TokenizerJsonArtifacts,
        runtime_configuration: RuntimeConfiguration,
    ) -> Result<Self, CoreError> {
        let metadata = Self::metadata_from_artifacts(&artifacts, runtime_configuration);
        let llama_post_processor_compatibility = Self::llama_post_processor_compatibility(
            &artifacts.tokenizer,
            &artifacts.metadata,
            &metadata.runtime_configuration,
        )?;
        Ok(Self {
            tokenizer: artifacts.tokenizer,
            environment: template::make_environment(),
            metadata,
            llama_post_processor_compatibility,
        })
    }

    fn metadata_from_artifacts(
        artifacts: &TokenizerJsonArtifacts,
        runtime_configuration: RuntimeConfiguration,
    ) -> TokenizerMetadata {
        TokenizerMetadata {
            bos_token_id: resolve_token_id(&artifacts.tokenizer, runtime_configuration.bos_token.as_deref()),
            eos_token_id: resolve_token_id(&artifacts.tokenizer, runtime_configuration.eos_token.as_deref()),
            unknown_token_id: resolve_token_id(
                &artifacts.tokenizer,
                runtime_configuration.unknown_token.as_deref()
            ),
            runtime_configuration,
        }
    }

    fn llama_post_processor_compatibility(
        tokenizer: &Tokenizer,
        tokenizer_json_metadata: &tokenizer_json::TokenizerJsonMetadata,
        runtime_configuration: &RuntimeConfiguration,
    ) -> Result<Option<LlamaPostProcessorCompatibility>, CoreError> {
        let tokenizer_class = runtime_configuration
            .tokenizer_class
            .as_deref()
            .map(normalized_tokenizer_name);
        if tokenizer_class.as_deref() != Some("LlamaTokenizer") {
            return Ok(None);
        }

        // Keep Rust aligned with the Swift backend and Python transformers. The
        // corresponding upstream path is `TokenizersBackend.update_post_processor`
        // in `tokenization_utils_tokenizers.py`, which rebuilds `TemplateProcessing`
        // from `add_bos_token` / `add_eos_token` for Llama-family fast tokenizers.
        if tokenizer_json_metadata.post_processor_type.as_deref() == Some("TemplateProcessing") {
            return Ok(None);
        }

        let bos_token_id = resolve_optional_compatibility_token_id(
            tokenizer,
            runtime_configuration.add_bos_token,
            runtime_configuration.bos_token.as_deref(),
            "add_bos_token is True but bos_token is nil",
            "bos_token is not present in the tokenizer vocabulary",
        )?;
        let eos_token_id = resolve_optional_compatibility_token_id(
            tokenizer,
            runtime_configuration.add_eos_token,
            runtime_configuration.eos_token.as_deref(),
            "add_eos_token is True but eos_token is nil",
            "eos_token is not present in the tokenizer vocabulary",
        )?;

        Ok(Some(LlamaPostProcessorCompatibility {
            bos_token_id,
            eos_token_id,
            add_bos_token: runtime_configuration.add_bos_token,
            add_eos_token: runtime_configuration.add_eos_token,
        }))
    }

    pub(crate) fn tokenize(&self, text: &str) -> Result<Vec<String>, CoreError> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| CoreError::Internal(err.to_string()))?;
        Ok(encoding.get_tokens().to_vec())
    }

    pub(crate) fn encode(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<i32>, CoreError> {
        let encoding = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|err| CoreError::Internal(err.to_string()))?;
        let mut token_ids: Vec<i32> = encoding.get_ids().iter().map(|id| *id as i32).collect();
        if add_special_tokens {
            if let Some(compatibility) = &self.llama_post_processor_compatibility {
                if compatibility.add_bos_token {
                    if let Some(bos_token_id) = compatibility.bos_token_id {
                        token_ids.insert(0, bos_token_id);
                    }
                }
                if compatibility.add_eos_token {
                    if let Some(eos_token_id) = compatibility.eos_token_id {
                        token_ids.push(eos_token_id);
                    }
                }
            }
        }
        Ok(token_ids)
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
        self.tokenizer.token_to_id(token).map(|token_id| token_id as i32)
    }

    pub(crate) fn convert_id_to_token(&self, token_id: i32) -> Option<String> {
        u32::try_from(token_id)
            .ok()
            .and_then(|token_id| self.tokenizer.id_to_token(token_id))
    }

    pub(crate) fn vocab_count(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub(crate) fn apply_chat_template(
        &self,
        template: &str,
        context: JsonValue,
        truncation: bool,
        max_length: Option<usize>,
    ) -> Result<Vec<i32>, CoreError> {
        let rendered = template::render(&self.environment, template, &context)?;
        let mut encoded = self.encode(&rendered, false)?;

        if truncation {
            if let Some(max_length) = max_length {
                if encoded.len() > max_length {
                    encoded.truncate(max_length);
                }
            }
        }

        Ok(encoded)
    }
}

fn resolve_token_id(tokenizer: &Tokenizer, token: Option<&str>) -> Option<i32> {
    token
        .and_then(|token| tokenizer.token_to_id(token))
        .map(|token_id| token_id as i32)
}

fn resolve_optional_compatibility_token_id(
    tokenizer: &Tokenizer,
    enabled: bool,
    token: Option<&str>,
    missing_token_message: &str,
    missing_id_message: &str,
) -> Result<Option<i32>, CoreError> {
    if !enabled {
        return Ok(None);
    }

    let token = token.ok_or_else(|| CoreError::MismatchedConfig(missing_token_message.to_owned()))?;
    resolve_token_id(tokenizer, Some(token))
        .ok_or_else(|| CoreError::MismatchedConfig(missing_id_message.to_owned()))
        .map(Some)
}

fn normalized_tokenizer_name(tokenizer_class: &str) -> String {
    tokenizer_class.replace("Fast", "")
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
    fn creates_tokenizer_core_from_offline_fixture_directory() {
        let directory = offline_fixture_directory();
        let core = TokenizerCore::from_directory(&directory)
            .unwrap_or_else(|error| panic!("failed to build tokenizer core: {error}"));

        assert_eq!(core.convert_token_to_id("<unk>"), Some(3));
        assert_eq!(core.metadata.runtime_configuration.unknown_token.as_deref(), Some("<unk>"));
    }
}
