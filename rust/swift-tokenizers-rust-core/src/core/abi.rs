use serde_json::Value as JsonValue;
use std::ffi::{c_char, CStr};
use std::mem;
use std::ptr;
use std::slice;

use crate::core::error::CoreError;
use crate::core::sidecars;
use crate::core::tokenizer_core::TokenizerCore;
use crate::core::tokenizer_json;

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_owned_buffer_t {
    data: *mut u8,
    len: usize,
}

#[allow(non_camel_case_types)]
#[repr(C)]
pub struct st_error_t {
    code: i32,
    message: st_owned_buffer_t,
}

#[allow(non_camel_case_types)]
pub struct st_tokenizer_handle {
    core: TokenizerCore,
}

fn empty_buffer() -> st_owned_buffer_t {
    st_owned_buffer_t {
        data: ptr::null_mut(),
        len: 0,
    }
}

fn vec_to_buffer(mut bytes: Vec<u8>) -> st_owned_buffer_t {
    let buffer = st_owned_buffer_t {
        data: bytes.as_mut_ptr(),
        len: bytes.len(),
    };
    mem::forget(bytes);
    buffer
}

fn string_to_buffer(value: String) -> st_owned_buffer_t {
    vec_to_buffer(value.into_bytes())
}

fn clear_error(out_error: *mut st_error_t) {
    if out_error.is_null() {
        return;
    }

    unsafe {
        (*out_error).code = 0;
        (*out_error).message = empty_buffer();
    }
}

fn write_error(out_error: *mut st_error_t, error: CoreError) {
    if out_error.is_null() {
        return;
    }

    unsafe {
        (*out_error).code = error.code();
        (*out_error).message = string_to_buffer(error.to_string());
    }
}

fn read_required_utf8(value: *const c_char, field: &str) -> Result<String, CoreError> {
    if value.is_null() {
        return Err(CoreError::Internal(format!("{field} was null")));
    }

    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map(|value| value.to_owned())
        .map_err(|_| CoreError::Internal(format!("{field} was not valid UTF-8")))
}

fn read_json_arg(value: *const c_char, field: &str) -> Result<JsonValue, CoreError> {
    let json = read_required_utf8(value, field)?;
    serde_json::from_str(&json).map_err(|err| CoreError::MismatchedConfig(err.to_string()))
}

fn write_metadata_json(handle: &st_tokenizer_handle) -> Result<st_owned_buffer_t, CoreError> {
    serde_json::to_vec(&handle.core.metadata)
        .map(vec_to_buffer)
        .map_err(|err| CoreError::Internal(err.to_string()))
}

fn write_json_buffer<T: serde::Serialize>(value: &T) -> Result<st_owned_buffer_t, CoreError> {
    serde_json::to_vec(value)
        .map(vec_to_buffer)
        .map_err(|err| CoreError::Internal(err.to_string()))
}

fn write_i32_output(values: Vec<i32>, out_token_ids: *mut *mut i32, out_len: *mut usize) {
    if out_token_ids.is_null() || out_len.is_null() {
        return;
    }

    let mut values = values;
    unsafe {
        *out_len = values.len();
        *out_token_ids = values.as_mut_ptr();
    }
    mem::forget(values);
}

fn success_no_value(out_error: *mut st_error_t) -> bool {
    clear_error(out_error);
    true
}

fn failure(out_error: *mut st_error_t, error: CoreError) -> bool {
    write_error(out_error, error);
    false
}

#[no_mangle]
pub extern "C" fn st_tokenizer_create_from_directory(
    directory_path: *const c_char,
    out_handle: *mut *mut st_tokenizer_handle,
    out_metadata_json: *mut st_owned_buffer_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    let directory_path = match read_required_utf8(directory_path, "directory_path") {
        Ok(directory_path) => directory_path,
        Err(error) => return failure(out_error, error),
    };

    let core = match TokenizerCore::from_directory(std::path::Path::new(&directory_path)) {
        Ok(core) => core,
        Err(error) => return failure(out_error, error),
    };

    let handle = Box::new(st_tokenizer_handle { core });
    let metadata_json = match write_metadata_json(&handle) {
        Ok(metadata_json) => metadata_json,
        Err(error) => return failure(out_error, error),
    };

    if out_handle.is_null() || out_metadata_json.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    unsafe {
        *out_metadata_json = metadata_json;
        *out_handle = Box::into_raw(handle);
    }

    true
}

#[no_mangle]
pub extern "C" fn st_load_tokenizer_runtime_configuration(
    directory_path: *const c_char,
    out_runtime_configuration_json: *mut st_owned_buffer_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if out_runtime_configuration_json.is_null() {
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
    let runtime_configuration_json = match write_json_buffer(&runtime_configuration) {
        Ok(runtime_configuration_json) => runtime_configuration_json,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_runtime_configuration_json = runtime_configuration_json;
    }

    true
}

#[no_mangle]
pub extern "C" fn st_tokenizer_create_from_tokenizer_json(
    directory_path: *const c_char,
    runtime_configuration_json: *const c_char,
    out_handle: *mut *mut st_tokenizer_handle,
    out_metadata_json: *mut st_owned_buffer_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    let directory_path = match read_required_utf8(directory_path, "directory_path") {
        Ok(directory_path) => directory_path,
        Err(error) => return failure(out_error, error),
    };

    let runtime_configuration = match read_json_arg(runtime_configuration_json, "runtime_configuration_json")
        .and_then(|value| {
            serde_json::from_value(value).map_err(|err| CoreError::MismatchedConfig(err.to_string()))
        }) {
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
    let metadata_json = match write_metadata_json(&handle) {
        Ok(metadata_json) => metadata_json,
        Err(error) => return failure(out_error, error),
    };

    if out_handle.is_null() || out_metadata_json.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    unsafe {
        *out_metadata_json = metadata_json;
        *out_handle = Box::into_raw(handle);
    }

    true
}

#[no_mangle]
pub extern "C" fn st_tokenizer_destroy(handle: *mut st_tokenizer_handle) {
    if handle.is_null() {
        return;
    }

    unsafe {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub extern "C" fn st_tokenizer_tokenize_to_json(
    handle: *const st_tokenizer_handle,
    text: *const c_char,
    out_tokens_json: *mut st_owned_buffer_t,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() || out_tokens_json.is_null() {
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

    let payload = match serde_json::to_vec(&tokens) {
        Ok(payload) => payload,
        Err(error) => return failure(out_error, CoreError::Internal(error.to_string())),
    };

    unsafe {
        *out_tokens_json = vec_to_buffer(payload);
    }

    true
}

#[no_mangle]
pub extern "C" fn st_tokenizer_encode(
    handle: *const st_tokenizer_handle,
    text: *const c_char,
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

    let values = match unsafe { &*handle }.core.encode(&text, add_special_tokens) {
        Ok(values) => values,
        Err(error) => return failure(out_error, error),
    };

    write_i32_output(values, out_token_ids, out_len);
    success_no_value(out_error)
}

#[no_mangle]
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
        return failure(out_error, CoreError::Internal("token_ids was null".to_owned()));
    } else {
        unsafe { slice::from_raw_parts(token_ids, len) }
    };

    let decoded = match unsafe { &*handle }.core.decode(token_ids, skip_special_tokens) {
        Ok(decoded) => decoded,
        Err(error) => return failure(out_error, error),
    };

    unsafe {
        *out_text = string_to_buffer(decoded);
    }

    true
}

#[no_mangle]
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

#[no_mangle]
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
        *out_token = maybe_token.map(string_to_buffer).unwrap_or_else(empty_buffer);
    }

    true
}

#[no_mangle]
pub extern "C" fn st_tokenizer_vocab_count(
    handle: *const st_tokenizer_handle,
    out_vocab_count: *mut usize,
    out_error: *mut st_error_t,
) -> bool {
    // This stays separate from tokenizer creation metadata on purpose: unlike the Swift backend,
    // the Rust path follows Python fast tokenizers and resolves vocab size lazily so it does not
    // add avoidable work to the tokenizer load path.
    clear_error(out_error);

    if handle.is_null() || out_vocab_count.is_null() {
        return failure(
            out_error,
            CoreError::Internal("output pointers were null".to_owned()),
        );
    }

    unsafe {
        *out_vocab_count = (&*handle).core.vocab_count();
    }

    true
}

#[no_mangle]
pub extern "C" fn st_tokenizer_apply_chat_template(
    handle: *const st_tokenizer_handle,
    template: *const c_char,
    context_json: *const c_char,
    truncation: bool,
    has_max_length: bool,
    max_length: u32,
    out_token_ids: *mut *mut i32,
    out_len: *mut usize,
    out_error: *mut st_error_t,
) -> bool {
    clear_error(out_error);

    if handle.is_null() {
        return failure(out_error, CoreError::Internal("handle was null".to_owned()));
    }

    let template = match read_required_utf8(template, "template") {
        Ok(template) => template,
        Err(error) => return failure(out_error, error),
    };
    let context = match read_json_arg(context_json, "context_json") {
        Ok(context) => context,
        Err(error) => return failure(out_error, error),
    };

    let max_length = if has_max_length {
        Some(max_length as usize)
    } else {
        None
    };

    let values = match unsafe { &*handle }.core.apply_chat_template(
        &template,
        context,
        truncation,
        max_length,
    ) {
        Ok(values) => values,
        Err(error) => return failure(out_error, error),
    };

    write_i32_output(values, out_token_ids, out_len);
    success_no_value(out_error)
}

#[no_mangle]
pub extern "C" fn st_render_template(
    template: *const c_char,
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

    let template = match read_required_utf8(template, "template") {
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

#[no_mangle]
pub extern "C" fn st_free_owned_buffer(buffer: st_owned_buffer_t) {
    if buffer.data.is_null() || buffer.len == 0 {
        return;
    }

    unsafe {
        drop(Vec::from_raw_parts(buffer.data, buffer.len, buffer.len));
    }
}

#[no_mangle]
pub extern "C" fn st_free_int32_array(data: *mut i32, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }

    unsafe {
        drop(Vec::from_raw_parts(data, len, len));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::path::PathBuf;

    fn offline_fixture_directory() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../Tests/TokenizersTests/Resources")
    }

    #[test]
    fn creates_tokenizer_handle_from_directory_via_abi() {
        let directory = CString::new(offline_fixture_directory().to_string_lossy().into_owned())
            .expect("fixture path should be valid C string");
        let mut handle: *mut st_tokenizer_handle = ptr::null_mut();
        let mut metadata = empty_buffer();
        let mut error = st_error_t {
            code: 0,
            message: empty_buffer(),
        };

        let success = st_tokenizer_create_from_directory(
            directory.as_ptr(),
            &mut handle,
            &mut metadata,
            &mut error,
        );

        assert!(success, "ABI load failed with code {}: {}", error.code, string_to_buffer_lossy(&error.message));
        assert!(!handle.is_null());

        st_free_owned_buffer(metadata);
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
