// Copyright © Anthony DePasquale

#[derive(Debug, thiserror::Error)]
pub(crate) enum CoreError {
    #[error("Tokenizer configuration is missing.")]
    MissingConfig,
    #[error("Chat template error: {0}")]
    ChatTemplate(String),
    #[error("Tokenizer configuration mismatch: {0}")]
    MismatchedConfig(String),
    #[error("{0}")]
    Internal(String),
}
