use std::mem;
use std::ptr;
use std::slice;

use crate::core::error::CoreError;

use super::types::{st_named_chat_template_t, st_owned_buffer_t, st_owned_int32_array_t};

pub(crate) fn empty_buffer() -> st_owned_buffer_t {
    st_owned_buffer_t {
        data: ptr::null_mut(),
        len: 0,
    }
}

fn vec_to_buffer(bytes: Vec<u8>) -> st_owned_buffer_t {
    let (data, len) = vec_to_raw_parts(bytes);
    st_owned_buffer_t { data, len }
}

pub(crate) fn string_to_buffer(value: String) -> st_owned_buffer_t {
    vec_to_buffer(value.into_bytes())
}

pub(crate) fn write_i32_output(
    values: Vec<i32>,
    out_token_ids: *mut *mut i32,
    out_len: *mut usize,
) {
    if out_token_ids.is_null() || out_len.is_null() {
        return;
    }

    let (values, values_len) = vec_to_raw_parts(values);
    unsafe {
        *out_len = values_len;
        *out_token_ids = values;
    }
}

pub(crate) fn vec_to_raw_parts<T>(values: Vec<T>) -> (*mut T, usize) {
    if values.is_empty() {
        return (ptr::null_mut(), 0);
    }

    let mut values = values.into_boxed_slice();
    let pointer = values.as_mut_ptr();
    let len = values.len();
    mem::forget(values);
    (pointer, len)
}

pub(crate) fn optional_i32_array(values: Vec<Option<i32>>) -> (*mut i32, *mut bool, usize) {
    let mut raw_values = Vec::with_capacity(values.len());
    let mut present = Vec::with_capacity(values.len());
    for value in values {
        match value {
            Some(value) => {
                raw_values.push(value);
                present.push(true);
            }
            None => {
                raw_values.push(i32::MIN);
                present.push(false);
            }
        }
    }

    let (values_pointer, values_len) = vec_to_raw_parts(raw_values);
    let (present_pointer, present_len) = vec_to_raw_parts(present);
    debug_assert_eq!(values_len, present_len);
    (values_pointer, present_pointer, values_len)
}

pub(crate) fn optional_i32(value: Option<i32>) -> (i32, bool) {
    match value {
        Some(value) => (value, true),
        None => (0, false),
    }
}

pub(crate) fn buffer_to_string(
    buffer: &st_owned_buffer_t,
    field: &str,
) -> Result<String, CoreError> {
    if buffer.len == 0 {
        return Ok(String::new());
    }
    if buffer.data.is_null() {
        return Err(CoreError::Internal(format!("{field} was null")));
    }

    let bytes = unsafe { slice::from_raw_parts(buffer.data, buffer.len) };
    String::from_utf8(bytes.to_vec())
        .map_err(|_| CoreError::Internal(format!("{field} was not valid UTF-8")))
}

pub(crate) fn string_array_from_buffers(
    buffers: *const st_owned_buffer_t,
    len: usize,
    field: &str,
) -> Result<Vec<String>, CoreError> {
    if len == 0 {
        return Ok(Vec::new());
    }
    if buffers.is_null() {
        return Err(CoreError::Internal(format!("{field} was null")));
    }

    unsafe { slice::from_raw_parts(buffers, len) }
        .iter()
        .enumerate()
        .map(|(index, buffer)| buffer_to_string(buffer, &format!("{field}[{index}]")))
        .collect()
}

pub(crate) unsafe fn free_raw_array<T>(data: *mut T, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }

    let slice_pointer = ptr::slice_from_raw_parts_mut(data, len);
    unsafe {
        drop(Box::from_raw(slice_pointer));
    }
}

pub(crate) unsafe fn free_owned_buffer_slice(data: *mut st_owned_buffer_t, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }

    let slice_pointer = ptr::slice_from_raw_parts_mut(data, len);
    let buffers = unsafe { &mut *slice_pointer };
    for buffer in buffers {
        let buffer = mem::replace(buffer, empty_buffer());
        unsafe {
            free_raw_array(buffer.data, buffer.len);
        }
    }
    unsafe {
        drop(Box::from_raw(slice_pointer));
    }
}

pub(crate) unsafe fn free_owned_int32_array_slice(data: *mut st_owned_int32_array_t, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }

    let slice_pointer = ptr::slice_from_raw_parts_mut(data, len);
    let arrays = unsafe { &mut *slice_pointer };
    for array in arrays {
        let owned = mem::replace(
            array,
            st_owned_int32_array_t {
                data: ptr::null_mut(),
                len: 0,
            },
        );
        unsafe {
            free_raw_array(owned.data, owned.len);
        }
    }
    unsafe {
        drop(Box::from_raw(slice_pointer));
    }
}

pub(crate) unsafe fn free_named_chat_template_slice(
    data: *mut st_named_chat_template_t,
    len: usize,
) {
    if data.is_null() || len == 0 {
        return;
    }

    let slice_pointer = ptr::slice_from_raw_parts_mut(data, len);
    let templates = unsafe { &mut *slice_pointer };
    for template in templates {
        let name = mem::replace(&mut template.name, empty_buffer());
        let template_text = mem::replace(&mut template.template_text, empty_buffer());
        unsafe {
            free_raw_array(name.data, name.len);
            free_raw_array(template_text.data, template_text.len);
        }
    }
    unsafe {
        drop(Box::from_raw(slice_pointer));
    }
}
