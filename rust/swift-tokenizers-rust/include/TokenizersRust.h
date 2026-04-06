#ifndef TOKENIZERS_RUST_H
#define TOKENIZERS_RUST_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_tokenizer_handle st_tokenizer_handle_t;

typedef struct {
    uint8_t *data;
    size_t len;
} st_owned_buffer_t;

typedef struct {
    int32_t code;
    st_owned_buffer_t message;
} st_error_t;

bool st_tokenizer_create_from_directory(
    const char *directory_path,
    st_tokenizer_handle_t **out_handle,
    st_owned_buffer_t *out_metadata_json,
    st_error_t *out_error
);

bool st_load_tokenizer_runtime_configuration(
    const char *directory_path,
    st_owned_buffer_t *out_runtime_configuration_json,
    st_error_t *out_error
);

bool st_tokenizer_create_from_tokenizer_json(
    const char *directory_path,
    const char *runtime_configuration_json,
    st_tokenizer_handle_t **out_handle,
    st_owned_buffer_t *out_metadata_json,
    st_error_t *out_error
);

void st_tokenizer_destroy(st_tokenizer_handle_t *handle);

bool st_tokenizer_tokenize_to_json(
    const st_tokenizer_handle_t *handle,
    const char *text,
    st_owned_buffer_t *out_tokens_json,
    st_error_t *out_error
);

bool st_tokenizer_encode(
    const st_tokenizer_handle_t *handle,
    const char *text,
    bool add_special_tokens,
    int32_t **out_token_ids,
    size_t *out_len,
    st_error_t *out_error
);

bool st_tokenizer_decode(
    const st_tokenizer_handle_t *handle,
    const int32_t *token_ids,
    size_t len,
    bool skip_special_tokens,
    st_owned_buffer_t *out_text,
    st_error_t *out_error
);

bool st_tokenizer_convert_token_to_id(
    const st_tokenizer_handle_t *handle,
    const char *token,
    bool *out_found,
    int32_t *out_token_id,
    st_error_t *out_error
);

bool st_tokenizer_convert_id_to_token(
    const st_tokenizer_handle_t *handle,
    int32_t token_id,
    bool *out_found,
    st_owned_buffer_t *out_token,
    st_error_t *out_error
);

bool st_tokenizer_vocab_count(
    const st_tokenizer_handle_t *handle,
    size_t *out_vocab_count,
    st_error_t *out_error
);

bool st_tokenizer_apply_chat_template(
    const st_tokenizer_handle_t *handle,
    const char *template_text,
    const char *context_json,
    bool truncation,
    bool has_max_length,
    uint32_t max_length,
    int32_t **out_token_ids,
    size_t *out_len,
    st_error_t *out_error
);

bool st_render_template(
    const char *template_text,
    const char *context_json,
    st_owned_buffer_t *out_text,
    st_error_t *out_error
);

void st_free_owned_buffer(st_owned_buffer_t buffer);
void st_free_int32_array(int32_t *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif
