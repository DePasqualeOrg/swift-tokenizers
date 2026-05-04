//! C ABI used by Swift to call the Rust tokenizer backend.
//!
//! Ownership conventions:
//! - Input pointers are borrowed for the duration of the call.
//! - Output buffers and arrays are allocated by Rust and must be released by the matching `st_free_*` function.
//! - A null data pointer is valid only when its companion length is zero.
//! - Optional scalar and string fields use explicit `has_*` companion fields so empty values stay distinct from absent values.
//! - Optional integer arrays use parallel value and presence arrays; values are meaningful only when the matching presence entry is true.

mod buffers;
mod config;
mod encoding;
mod error;
mod types;

use serde_json::Value as JsonValue;
use std::ffi::{CStr, c_char};
use std::slice;

use crate::core::error::CoreError;
use crate::core::sidecars;
use crate::core::tokenizer_core::{EncodingOffsetUnit, TokenizerCore};
use crate::core::tokenizer_json;

use buffers::{
    empty_buffer, free_named_chat_template_slice, free_owned_buffer_slice,
    free_owned_int32_array_slice, free_raw_array, string_to_buffer, vec_to_raw_parts,
    write_i32_output,
};
#[cfg(test)]
use config::empty_tokenizer_descriptor;
use config::{
    runtime_configuration_from_ffi, runtime_configuration_to_ffi, tokenizer_descriptor_to_ffi,
};
use encoding::{encoding_to_ffi, free_encoding_array};
use error::{clear_error, failure, success_no_value};
pub use types::{
    st_encode_input_t, st_encoding_t, st_error_t, st_owned_buffer_t, st_owned_int32_array_t,
    st_runtime_configuration_t, st_tokenizer_descriptor_t, st_tokenizer_handle,
};

fn read_required_utf8(value: *const c_char, field: &str) -> Result<String, CoreError> {
    if value.is_null() {
        return Err(CoreError::Internal(format!("{field} was null")));
    }

    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(|value| value.to_owned())
        .map_err(|_| CoreError::Internal(format!("{field} was not valid UTF-8")))
}

fn read_optional_utf8(value: *const c_char, field: &str) -> Result<Option<String>, CoreError> {
    if value.is_null() {
        return Ok(None);
    }

    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(|value| Some(value.to_owned()))
        .map_err(|_| CoreError::Internal(format!("{field} was not valid UTF-8")))
}

fn read_json_arg(value: *const c_char, field: &str) -> Result<JsonValue, CoreError> {
    let json = read_required_utf8(value, field)?;
    serde_json::from_str(&json)
        .map_err(|err| CoreError::Internal(format!("invalid {field} JSON: {err}")))
}

fn read_encode_input_array(
    inputs: *const st_encode_input_t,
    inputs_len: usize,
) -> Result<Vec<(String, Option<String>)>, CoreError> {
    if inputs_len == 0 {
        return Ok(Vec::new());
    }
    if inputs.is_null() {
        return Err(CoreError::Internal("inputs was null".to_owned()));
    }

    let slice = unsafe { slice::from_raw_parts(inputs, inputs_len) };
    slice
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let text = read_required_utf8(input.text, &format!("inputs[{index}].text"))?;
            let text_pair =
                read_optional_utf8(input.text_pair, &format!("inputs[{index}].text_pair"))?;
            Ok((text, text_pair))
        })
        .collect()
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_create_from_directory(
    directory_path: *const c_char,
    out_handle: *mut *mut st_tokenizer_handle,
    out_descriptor: *mut st_tokenizer_descriptor_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if out_handle.is_null() || out_descriptor.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let directory_path = match read_required_utf8(directory_path, "directory_path") {
        Ok(directory_path) => directory_path,
        Err(error) => return failure(out_error, error),
    };

    let core = match TokenizerCore::from_directory(std::path::Path::new(&directory_path)) {
        Ok(core) => core,
        Err(error) => return failure(out_error, error),
    };

    let handle = Box::new(st_tokenizer_handle { core });
    let descriptor = match tokenizer_descriptor_to_ffi(handle.core.metadata.clone()) {
        Ok(descriptor) => descriptor,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_descriptor = descriptor;
        *out_handle = Box::into_raw(handle);
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_load_tokenizer_runtime_configuration(
    directory_path: *const c_char,
    out_runtime_configuration: *mut st_runtime_configuration_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if out_runtime_configuration.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let directory_path = match read_required_utf8(directory_path, "directory_path") {
        Ok(directory_path) => directory_path,
        Err(error) => return failure(out_error, error),
    };

    let runtime_configuration =
        sidecars::load_runtime_configuration_only(std::path::Path::new(&directory_path));
    let runtime_configuration = match runtime_configuration_to_ffi(runtime_configuration) {
        Ok(runtime_configuration) => runtime_configuration,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_runtime_configuration = runtime_configuration;
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_create_with_runtime_configuration(
    directory_path: *const c_char,
    runtime_configuration: *const st_runtime_configuration_t,
    out_handle: *mut *mut st_tokenizer_handle,
    out_descriptor: *mut st_tokenizer_descriptor_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if runtime_configuration.is_null() || out_handle.is_null() || out_descriptor.is_null() {
        return failure(
            out_error,
            CoreError::Internal("input or output pointers were null".to_owned()),
        );
    }

    let directory_path = match read_required_utf8(directory_path, "directory_path") {
        Ok(directory_path) => directory_path,
        Err(error) => return failure(out_error, error),
    };

    let runtime_configuration =
        match runtime_configuration_from_ffi(unsafe { &*runtime_configuration }) {
            Ok(runtime_configuration) => runtime_configuration,
            Err(error) => return failure(out_error, error),
        };

    let artifacts = match tokenizer_json::load_artifacts(std::path::Path::new(&directory_path)) {
        Ok(artifacts) => artifacts,
        Err(error) => return failure(out_error, error),
    };
    let core = match TokenizerCore::from_artifacts_and_runtime_configuration(
        artifacts,
        runtime_configuration,
    ) {
        Ok(core) => core,
        Err(error) => return failure(out_error, error),
    };

    let handle = Box::new(st_tokenizer_handle { core });
    let descriptor = match tokenizer_descriptor_to_ffi(handle.core.metadata.clone()) {
        Ok(descriptor) => descriptor,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_descriptor = descriptor;
        *out_handle = Box::into_raw(handle);
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_destroy(handle: *mut st_tokenizer_handle) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_tokenize(
    handle: *const st_tokenizer_handle,
    text: *const c_char,
    out_tokens: *mut *mut st_owned_buffer_t,
    out_len: *mut usize,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_tokens.is_null() || out_len.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let text = match read_required_utf8(text, "text") {
        Ok(text) => text,
        Err(error) => return failure(out_error, error),
    };

    let tokens = match unsafe { &*handle }.core.tokenize(&text) {
        Ok(tokens) => tokens,
        Err(error) => return failure(out_error, error),
    };

    let (tokens, tokens_len) = vec_to_raw_parts(tokens.into_iter().map(string_to_buffer).collect());

    unsafe {
        *out_tokens = tokens;
        *out_len = tokens_len;
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_encode(
    handle: *const st_tokenizer_handle,
    text: *const c_char,
    text_pair: *const c_char,
    add_special_tokens: bool,
    out_token_ids: *mut *mut i32,
    out_len: *mut usize,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() {
        return failure(out_error, CoreError::Internal("handle was null".to_owned()));
    }

    let text = match read_required_utf8(text, "text") {
        Ok(text) => text,
        Err(error) => return failure(out_error, error),
    };
    let text_pair = match read_optional_utf8(text_pair, "text_pair") {
        Ok(value) => value,
        Err(error) => return failure(out_error, error),
    };

    let values =
        match unsafe { &*handle }
            .core
            .encode(&text, text_pair.as_deref(), add_special_tokens)
        {
            Ok(values) => values,
            Err(error) => return failure(out_error, error),
        };

    write_i32_output(values, out_token_ids, out_len);
    success_no_value(out_error)
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_encode_with_metadata(
    handle: *const st_tokenizer_handle,
    text: *const c_char,
    text_pair: *const c_char,
    add_special_tokens: bool,
    offset_unit: u8,
    out_encoding: *mut st_encoding_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_encoding.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let text = match read_required_utf8(text, "text") {
        Ok(text) => text,
        Err(error) => return failure(out_error, error),
    };
    let text_pair = match read_optional_utf8(text_pair, "text_pair") {
        Ok(value) => value,
        Err(error) => return failure(out_error, error),
    };
    let offset_unit = match EncodingOffsetUnit::try_from(offset_unit) {
        Ok(offset_unit) => offset_unit,
        Err(error) => return failure(out_error, error),
    };

    let encoding = match unsafe { &*handle }.core.encode_with_metadata(
        &text,
        text_pair.as_deref(),
        add_special_tokens,
        offset_unit,
    ) {
        Ok(encoding) => encoding,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_encoding = encoding_to_ffi(encoding);
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_encode_batch(
    handle: *const st_tokenizer_handle,
    inputs: *const st_encode_input_t,
    inputs_len: usize,
    add_special_tokens: bool,
    out_arrays: *mut *mut st_owned_int32_array_t,
    out_arrays_len: *mut usize,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_arrays.is_null() || out_arrays_len.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let parsed_inputs = match read_encode_input_array(inputs, inputs_len) {
        Ok(parsed) => parsed,
        Err(error) => return failure(out_error, error),
    };

    let batch = match unsafe { &*handle }
        .core
        .encode_batch(parsed_inputs, add_special_tokens)
    {
        Ok(values) => values,
        Err(error) => return failure(out_error, error),
    };

    let owned_arrays: Vec<st_owned_int32_array_t> = batch
        .into_iter()
        .map(|values| {
            let (data, len) = vec_to_raw_parts(values);
            st_owned_int32_array_t { data, len }
        })
        .collect();
    let (data, len) = vec_to_raw_parts(owned_arrays);
    unsafe {
        *out_arrays = data;
        *out_arrays_len = len;
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_encode_batch_with_metadata(
    handle: *const st_tokenizer_handle,
    inputs: *const st_encode_input_t,
    inputs_len: usize,
    add_special_tokens: bool,
    offset_unit: u8,
    out_encodings: *mut *mut st_encoding_t,
    out_encodings_len: *mut usize,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_encodings.is_null() || out_encodings_len.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let parsed_inputs = match read_encode_input_array(inputs, inputs_len) {
        Ok(parsed) => parsed,
        Err(error) => return failure(out_error, error),
    };
    let offset_unit = match EncodingOffsetUnit::try_from(offset_unit) {
        Ok(offset_unit) => offset_unit,
        Err(error) => return failure(out_error, error),
    };

    let batch = match unsafe { &*handle }.core.encode_batch_with_metadata(
        parsed_inputs,
        add_special_tokens,
        offset_unit,
    ) {
        Ok(values) => values,
        Err(error) => return failure(out_error, error),
    };

    let ffi_encodings: Vec<st_encoding_t> = batch.into_iter().map(encoding_to_ffi).collect();
    let (data, len) = vec_to_raw_parts(ffi_encodings);
    unsafe {
        *out_encodings = data;
        *out_encodings_len = len;
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_decode(
    handle: *const st_tokenizer_handle,
    token_ids: *const i32,
    len: usize,
    skip_special_tokens: bool,
    out_text: *mut st_owned_buffer_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_text.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let token_ids = if len == 0 {
        &[]
    } else if token_ids.is_null() {
        return failure(
            out_error,
            CoreError::Internal("token_ids was null".to_owned()),
        );
    } else {
        unsafe { slice::from_raw_parts(token_ids, len) }
    };

    let decoded = match unsafe { &*handle }
        .core
        .decode(token_ids, skip_special_tokens)
    {
        Ok(decoded) => decoded,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_text = string_to_buffer(decoded);
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_convert_token_to_id(
    handle: *const st_tokenizer_handle,
    token: *const c_char,
    out_found: *mut bool,
    out_token_id: *mut i32,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_found.is_null() || out_token_id.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let token = match read_required_utf8(token, "token") {
        Ok(token) => token,
        Err(error) => return failure(out_error, error),
    };

    let maybe_id = unsafe { &*handle }.core.convert_token_to_id(&token);

    unsafe {
        *out_found = maybe_id.is_some();
        *out_token_id = maybe_id.unwrap_or_default();
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_convert_id_to_token(
    handle: *const st_tokenizer_handle,
    token_id: i32,
    out_found: *mut bool,
    out_token: *mut st_owned_buffer_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_found.is_null() || out_token.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let maybe_token = unsafe { &*handle }.core.convert_id_to_token(token_id);

    unsafe {
        *out_found = maybe_token.is_some();
        *out_token = maybe_token
            .map(string_to_buffer)
            .unwrap_or_else(empty_buffer);
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_tokenizer_apply_chat_template(
    handle: *const st_tokenizer_handle,
    template_text: *const c_char,
    context_json: *const c_char,
    truncation: bool,
    has_max_length: bool,
    max_length: u64,
    out_token_ids: *mut *mut i32,
    out_len: *mut usize,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() {
        return failure(out_error, CoreError::Internal("handle was null".to_owned()));
    }

    let template = match read_required_utf8(template_text, "template_text") {
        Ok(template) => template,
        Err(error) => return failure(out_error, error),
    };
    let context = match read_json_arg(context_json, "context_json") {
        Ok(context) => context,
        Err(error) => return failure(out_error, error),
    };

    let max_length = if has_max_length {
        match usize::try_from(max_length) {
            Ok(value) => Some(value),
            Err(_) => {
                return failure(
                    out_error,
                    CoreError::Internal("max_length does not fit in usize".to_owned()),
                );
            }
        }
    } else {
        None
    };

    let values = match unsafe { &*handle }
        .core
        .apply_chat_template(&template, context, truncation, max_length)
    {
        Ok(values) => values,
        Err(error) => return failure(out_error, error),
    };

    write_i32_output(values, out_token_ids, out_len);
    success_no_value(out_error)
}

#[unsafe(no_mangle)]
pub extern "C" fn st_render_template(
    template_text: *const c_char,
    context_json: *const c_char,
    out_text: *mut st_owned_buffer_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if out_text.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    let template = match read_required_utf8(template_text, "template_text") {
        Ok(template) => template,
        Err(error) => return failure(out_error, error),
    };
    let context = match read_json_arg(context_json, "context_json") {
        Ok(context) => context,
        Err(error) => return failure(out_error, error),
    };

    let environment = crate::core::template::make_environment();
    let rendered = match crate::core::template::render(&environment, &template, &context) {
        Ok(rendered) => rendered,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_text = string_to_buffer(rendered);
    }

    true
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_owned_buffer(buffer: st_owned_buffer_t) {
    unsafe {
        free_raw_array(buffer.data, buffer.len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_owned_buffer_array(buffers: *mut st_owned_buffer_t, len: usize) {
    unsafe {
        free_owned_buffer_slice(buffers, len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_runtime_configuration(configuration: st_runtime_configuration_t) {
    st_free_owned_buffer(configuration.bos_token);
    st_free_owned_buffer(configuration.eos_token);
    st_free_owned_buffer(configuration.unknown_token);
    st_free_owned_buffer(configuration.sep_token);
    st_free_owned_buffer(configuration.pad_token);
    st_free_owned_buffer(configuration.cls_token);
    st_free_owned_buffer(configuration.mask_token);
    unsafe {
        free_owned_buffer_slice(
            configuration.additional_special_tokens,
            configuration.additional_special_tokens_len,
        );
    }
    st_free_owned_buffer(configuration.chat_template_literal);
    unsafe {
        free_named_chat_template_slice(
            configuration.named_chat_templates,
            configuration.named_chat_templates_len,
        );
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_tokenizer_descriptor(descriptor: st_tokenizer_descriptor_t) {
    st_free_runtime_configuration(descriptor.runtime_configuration);
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_int32_array(data: *mut i32, len: usize) {
    unsafe {
        free_raw_array(data, len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_owned_int32_array_array(data: *mut st_owned_int32_array_t, len: usize) {
    unsafe {
        free_owned_int32_array_slice(data, len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_encoding_array(data: *mut st_encoding_t, len: usize) {
    unsafe {
        free_encoding_array(data, len);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn st_free_encoding(encoding: st_encoding_t) {
    st_free_int32_array(encoding.token_ids, encoding.token_ids_len);
    st_free_int32_array(encoding.token_type_ids, encoding.token_type_ids_len);
    unsafe {
        free_owned_buffer_slice(encoding.tokens, encoding.tokens_len);
    }
    st_free_int32_array(encoding.word_indices, encoding.word_indices_len);
    unsafe {
        free_raw_array(encoding.word_indices_present, encoding.word_indices_len);
        free_raw_array(encoding.offset_spans, encoding.offset_spans_len);
    }
    st_free_int32_array(encoding.special_token_mask, encoding.special_token_mask_len);
    st_free_int32_array(encoding.attention_mask, encoding.attention_mask_len);
    st_free_int32_array(encoding.sequence_indices, encoding.sequence_indices_len);
    unsafe {
        free_raw_array(
            encoding.sequence_indices_present,
            encoding.sequence_indices_len,
        );
        free_encoding_array(encoding.overflow_encodings, encoding.overflow_encodings_len);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::path::PathBuf;
    use std::ptr;

    fn offline_fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Tests/TokenizersTests/Resources")
    }

    #[test]
    fn creates_tokenizer_handle_from_directory_via_abi() {
        let directory = CString::new(offline_fixture_directory().to_string_lossy().into_owned())
            .expect("fixture path should be valid C string");
        let mut handle: *mut st_tokenizer_handle = ptr::null_mut();
        let mut descriptor = empty_tokenizer_descriptor();
        let mut error = st_error_t {
            code: 0,
            message: empty_buffer(),
        };

        let success = st_tokenizer_create_from_directory(
            directory.as_ptr(),
            &mut handle,
            &mut descriptor,
            &mut error,
        );

        assert!(
            success,
            "ABI load failed with code {}: {}",
            error.code,
            string_to_buffer_lossy(&error.message)
        );
        assert!(!handle.is_null());
        assert!(descriptor.runtime_configuration.has_unknown_token);

        st_free_tokenizer_descriptor(descriptor);
        st_tokenizer_destroy(handle);
    }

    fn string_to_buffer_lossy(buffer: &st_owned_buffer_t) -> String {
        if buffer.data.is_null() || buffer.len == 0 {
            return String::new();
        }

        let bytes = unsafe { slice::from_raw_parts(buffer.data, buffer.len) };
        String::from_utf8_lossy(bytes).into_owned()
    }
}
