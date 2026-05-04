use crate::core::error::CoreError;

use super::buffers::{empty_buffer, string_to_buffer};
use super::types::st_error_t;

pub(crate) fn clear_error(out_error: *mut st_error_t) {
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
        (*out_error).message = string_to_buffer(error.ffi_message().unwrap_or_default());
    }
}

pub(crate) fn success_no_value(out_error: *mut st_error_t) -> bool {
    clear_error(out_error);
    true
}

pub(crate) fn failure(out_error: *mut st_error_t, error: CoreError) -> bool {
    write_error(out_error, error);
    false
}
