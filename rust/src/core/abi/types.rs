use crate::core::tokenizer_core::TokenizerCore;

// Optional string fields use a `buffer + has_buffer` pair instead of a nullable
// pointer because the buffer alone cannot distinguish "absent" from "empty
// string" — both encode as `data: null, len: 0`. The `has_*` companion flag
// disambiguates, matching the Swift `Optional<String>` shape exactly.

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_owned_buffer_t {
    pub(crate) data: *mut u8,
    pub(crate) len: usize,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_error_t {
    pub(crate) code: i32,
    pub(crate) message: st_owned_buffer_t,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_named_chat_template_t {
    pub(crate) name: st_owned_buffer_t,
    pub(crate) template_text: st_owned_buffer_t,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_runtime_configuration_t {
    pub(crate) bos_token: st_owned_buffer_t,
    pub(crate) has_bos_token: bool,
    pub(crate) eos_token: st_owned_buffer_t,
    pub(crate) has_eos_token: bool,
    pub(crate) unknown_token: st_owned_buffer_t,
    pub(crate) has_unknown_token: bool,
    pub(crate) sep_token: st_owned_buffer_t,
    pub(crate) has_sep_token: bool,
    pub(crate) pad_token: st_owned_buffer_t,
    pub(crate) has_pad_token: bool,
    pub(crate) cls_token: st_owned_buffer_t,
    pub(crate) has_cls_token: bool,
    pub(crate) mask_token: st_owned_buffer_t,
    pub(crate) has_mask_token: bool,
    pub(crate) additional_special_tokens: *mut st_owned_buffer_t,
    pub(crate) additional_special_tokens_len: usize,
    pub(crate) clean_up_tokenization_spaces: bool,
    pub(crate) model_max_length: u64,
    pub(crate) has_model_max_length: bool,
    pub(crate) chat_template_kind: u8,
    pub(crate) chat_template_literal: st_owned_buffer_t,
    pub(crate) named_chat_templates: *mut st_named_chat_template_t,
    pub(crate) named_chat_templates_len: usize,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_tokenizer_descriptor_t {
    pub(crate) runtime_configuration: st_runtime_configuration_t,
    pub(crate) bos_token_id: i32,
    pub(crate) has_bos_token_id: bool,
    pub(crate) eos_token_id: i32,
    pub(crate) has_eos_token_id: bool,
    pub(crate) unknown_token_id: i32,
    pub(crate) has_unknown_token_id: bool,
    pub(crate) base_vocab_size: usize,
    pub(crate) total_vocab_size: usize,
}

/// One element of a batch encode input: a primary text plus an optional pair sequence.
///
/// `text` must be non-null. `text_pair` is null for single-sequence input or non-null for
/// dual-sequence (sentence-pair) input. Mirrors `tokenizers::EncodeInput::{Single, Dual}`.
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_encode_input_t {
    pub(crate) text: *const std::os::raw::c_char,
    pub(crate) text_pair: *const std::os::raw::c_char,
}

/// One element in a batch fast-encode result: a heap-allocated `i32` array owned by the
/// callee. Free with `st_free_int32_array_array`.
#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_owned_int32_array_t {
    pub(crate) data: *mut i32,
    pub(crate) len: usize,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_encoding_offset_span_t {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_encoding_t {
    pub(crate) token_ids: *mut i32,
    pub(crate) token_ids_len: usize,
    pub(crate) token_type_ids: *mut i32,
    pub(crate) token_type_ids_len: usize,
    pub(crate) tokens: *mut st_owned_buffer_t,
    pub(crate) tokens_len: usize,
    pub(crate) word_indices: *mut i32,
    pub(crate) word_indices_present: *mut bool,
    pub(crate) word_indices_len: usize,
    pub(crate) offset_spans: *mut st_encoding_offset_span_t,
    pub(crate) offset_spans_len: usize,
    pub(crate) special_token_mask: *mut i32,
    pub(crate) special_token_mask_len: usize,
    pub(crate) attention_mask: *mut i32,
    pub(crate) attention_mask_len: usize,
    pub(crate) sequence_indices: *mut i32,
    pub(crate) sequence_indices_present: *mut bool,
    pub(crate) sequence_indices_len: usize,
    pub(crate) sequence_count: usize,
    pub(crate) overflow_encodings: *mut st_encoding_t,
    pub(crate) overflow_encodings_len: usize,
    pub(crate) offset_unit: u8,
}

#[allow(non_camel_case_types)]
pub struct st_tokenizer_handle {
    pub(crate) core: TokenizerCore,
}
