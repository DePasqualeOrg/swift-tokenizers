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

impl CoreError {
    pub(crate) fn code(&self) -> i32 {
        match self {
            Self::MissingConfig => 1,
            Self::ChatTemplate(_) => 6,
            Self::MismatchedConfig(_) => 9,
            Self::Internal(_) => 100,
        }
    }
}
