use std::mem;
use std::ptr;

use crate::core::tokenizer_core::{EncodingMetadata, EncodingOffsetSpan};

use super::buffers::{optional_i32_array, string_to_buffer, vec_to_raw_parts};
use super::types::{st_encoding_offset_span_t, st_encoding_t};

pub(crate) fn empty_encoding() -> st_encoding_t {
    st_encoding_t {
        token_ids: ptr::null_mut(),
        token_ids_len: 0,
        token_type_ids: ptr::null_mut(),
        token_type_ids_len: 0,
        tokens: ptr::null_mut(),
        tokens_len: 0,
        word_indices: ptr::null_mut(),
        word_indices_present: ptr::null_mut(),
        word_indices_len: 0,
        offset_spans: ptr::null_mut(),
        offset_spans_len: 0,
        special_token_mask: ptr::null_mut(),
        special_token_mask_len: 0,
        attention_mask: ptr::null_mut(),
        attention_mask_len: 0,
        sequence_indices: ptr::null_mut(),
        sequence_indices_present: ptr::null_mut(),
        sequence_indices_len: 0,
        sequence_count: 0,
        overflow_encodings: ptr::null_mut(),
        overflow_encodings_len: 0,
        offset_unit: 0,
    }
}

pub(crate) fn encoding_to_ffi(encoding: EncodingMetadata) -> st_encoding_t {
    let EncodingMetadata {
        ids,
        type_ids,
        tokens,
        word_ids,
        offsets,
        special_tokens_mask,
        attention_mask,
        sequence_ids,
        n_sequences,
        overflowing,
        offset_unit,
    } = encoding;

    let (token_ids, token_ids_len) = vec_to_raw_parts(ids);
    let (token_type_ids, token_type_ids_len) = vec_to_raw_parts(type_ids);
    let (tokens, tokens_len) = vec_to_raw_parts(tokens.into_iter().map(string_to_buffer).collect());
    let (word_indices, word_indices_present, word_indices_len) = optional_i32_array(word_ids);
    let (offset_spans, offset_spans_len) = vec_to_raw_parts(
        offsets
            .into_iter()
            .map(|EncodingOffsetSpan { start, end }| st_encoding_offset_span_t { start, end })
            .collect(),
    );
    let (special_token_mask, special_token_mask_len) = vec_to_raw_parts(special_tokens_mask);
    let (attention_mask, attention_mask_len) = vec_to_raw_parts(attention_mask);
    let (sequence_indices, sequence_indices_present, sequence_indices_len) =
        optional_i32_array(sequence_ids);
    let (overflow_encodings, overflow_encodings_len) =
        vec_to_raw_parts(overflowing.into_iter().map(encoding_to_ffi).collect());

    st_encoding_t {
        token_ids,
        token_ids_len,
        token_type_ids,
        token_type_ids_len,
        tokens,
        tokens_len,
        word_indices,
        word_indices_present,
        word_indices_len,
        offset_spans,
        offset_spans_len,
        special_token_mask,
        special_token_mask_len,
        attention_mask,
        attention_mask_len,
        sequence_indices,
        sequence_indices_present,
        sequence_indices_len,
        sequence_count: n_sequences,
        overflow_encodings,
        overflow_encodings_len,
        offset_unit: offset_unit as u8,
    }
}

pub(crate) unsafe fn free_encoding_array(data: *mut st_encoding_t, len: usize) {
    if data.is_null() || len == 0 {
        return;
    }

    let slice_pointer = ptr::slice_from_raw_parts_mut(data, len);
    let encodings = unsafe { &mut *slice_pointer };
    for encoding in encodings {
        let encoding = mem::replace(encoding, empty_encoding());
        super::st_free_encoding(encoding);
    }
    unsafe {
        drop(Box::from_raw(slice_pointer));
    }
}
