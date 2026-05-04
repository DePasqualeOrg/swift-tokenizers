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

typedef struct {
    st_owned_buffer_t name;
    st_owned_buffer_t template_text;
} st_named_chat_template_t;

typedef struct {
    st_owned_buffer_t bos_token;
    bool has_bos_token;
    st_owned_buffer_t eos_token;
    bool has_eos_token;
    st_owned_buffer_t unknown_token;
    bool has_unknown_token;
    st_owned_buffer_t sep_token;
    bool has_sep_token;
    st_owned_buffer_t pad_token;
    bool has_pad_token;
    st_owned_buffer_t cls_token;
    bool has_cls_token;
    st_owned_buffer_t mask_token;
    bool has_mask_token;
    st_owned_buffer_t *additional_special_tokens;
    size_t additional_special_tokens_len;
    bool clean_up_tokenization_spaces;
    uint64_t model_max_length;
    bool has_model_max_length;
    uint8_t chat_template_kind;
    st_owned_buffer_t chat_template_literal;
    st_named_chat_template_t *named_chat_templates;
    size_t named_chat_templates_len;
} st_runtime_configuration_t;

typedef struct {
    st_runtime_configuration_t runtime_configuration;
    int32_t bos_token_id;
    bool has_bos_token_id;
    int32_t eos_token_id;
    bool has_eos_token_id;
    int32_t unknown_token_id;
    bool has_unknown_token_id;
    size_t base_vocab_size;
    size_t total_vocab_size;
} st_tokenizer_descriptor_t;

typedef struct {
    size_t start;
    size_t end;
} st_encoding_offset_span_t;

typedef struct st_encoding {
    int32_t *token_ids;
    size_t token_ids_len;
    int32_t *token_type_ids;
    size_t token_type_ids_len;
    st_owned_buffer_t *tokens;
    size_t tokens_len;
    int32_t *word_indices;
    bool *word_indices_present;
    size_t word_indices_len;
    st_encoding_offset_span_t *offset_spans;
    size_t offset_spans_len;
    int32_t *special_token_mask;
    size_t special_token_mask_len;
    int32_t *attention_mask;
    size_t attention_mask_len;
    int32_t *sequence_indices;
    bool *sequence_indices_present;
    size_t sequence_indices_len;
    size_t sequence_count;
    struct st_encoding *overflow_encodings;
    size_t overflow_encodings_len;
    uint8_t offset_unit;
} st_encoding_t;

bool st_tokenizer_create_from_directory(
    const char *directory_path,
    st_tokenizer_handle_t **out_handle,
    st_tokenizer_descriptor_t *out_descriptor,
    st_error_t *out_error
);

bool st_load_tokenizer_runtime_configuration(
    const char *directory_path,
    st_runtime_configuration_t *out_runtime_configuration,
    st_error_t *out_error
);

bool st_tokenizer_create_with_runtime_configuration(
    const char *directory_path,
    const st_runtime_configuration_t *runtime_configuration,
    st_tokenizer_handle_t **out_handle,
    st_tokenizer_descriptor_t *out_descriptor,
    st_error_t *out_error
);

void st_tokenizer_destroy(st_tokenizer_handle_t *handle);

bool st_tokenizer_tokenize(
    const st_tokenizer_handle_t *handle,
    const char *text,
    st_owned_buffer_t **out_tokens,
    size_t *out_len,
    st_error_t *out_error
);

bool st_tokenizer_encode(
    const st_tokenizer_handle_t *handle,
    const char *text,
    const char *text_pair,
    bool add_special_tokens,
    int32_t **out_token_ids,
    size_t *out_len,
    st_error_t *out_error
);

bool st_tokenizer_encode_with_metadata(
    const st_tokenizer_handle_t *handle,
    const char *text,
    const char *text_pair,
    bool add_special_tokens,
    uint8_t offset_unit,
    st_encoding_t *out_encoding,
    st_error_t *out_error
);

typedef struct {
    const char *text;
    const char *text_pair;
} st_encode_input_t;

typedef struct {
    int32_t *data;
    size_t len;
} st_owned_int32_array_t;

bool st_tokenizer_encode_batch(
    const st_tokenizer_handle_t *handle,
    const st_encode_input_t *inputs,
    size_t inputs_len,
    bool add_special_tokens,
    st_owned_int32_array_t **out_arrays,
    size_t *out_arrays_len,
    st_error_t *out_error
);

bool st_tokenizer_encode_batch_with_metadata(
    const st_tokenizer_handle_t *handle,
    const st_encode_input_t *inputs,
    size_t inputs_len,
    bool add_special_tokens,
    uint8_t offset_unit,
    st_encoding_t **out_encodings,
    size_t *out_encodings_len,
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

bool st_tokenizer_apply_chat_template(
    const st_tokenizer_handle_t *handle,
    const char *template_text,
    const char *context_json,
    bool truncation,
    bool has_max_length,
    uint64_t max_length,
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
void st_free_owned_buffer_array(st_owned_buffer_t *buffers, size_t len);
void st_free_runtime_configuration(st_runtime_configuration_t configuration);
void st_free_tokenizer_descriptor(st_tokenizer_descriptor_t descriptor);
void st_free_int32_array(int32_t *data, size_t len);
void st_free_owned_int32_array_array(st_owned_int32_array_t *data, size_t len);
void st_free_encoding(st_encoding_t encoding);
void st_free_encoding_array(st_encoding_t *data, size_t len);

#ifdef __cplusplus
}
#endif

#endif
