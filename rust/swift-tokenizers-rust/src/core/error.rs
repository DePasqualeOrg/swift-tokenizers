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

    /// The bare message payload sent across the FFI boundary, or `None` for
    /// variants without a payload. Excludes the category prefix that `Display`
    /// adds, since the Swift consumer's `TokenizerError.errorDescription` is the
    /// canonical place for that prefix.
    pub(crate) fn ffi_message(&self) -> Option<String> {
        match self {
            Self::MissingConfig => None,
            Self::ChatTemplate(s) | Self::MismatchedConfig(s) | Self::Internal(s) => {
                Some(s.clone())
            }
        }
    }
}
