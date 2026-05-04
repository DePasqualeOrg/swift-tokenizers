use serde_json::Value as JsonValue;
use std::ptr;
use std::slice;

use crate::core::error::CoreError;
use crate::core::sidecars::{RuntimeConfiguration, TokenizerMetadata};

use super::buffers::{
    buffer_to_string, empty_buffer, optional_i32, string_array_from_buffers, string_to_buffer,
    vec_to_raw_parts,
};
use super::types::{
    st_named_chat_template_t, st_owned_buffer_t, st_runtime_configuration_t,
    st_tokenizer_descriptor_t,
};

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FfiChatTemplateKind {
    None = 0,
    Literal = 1,
    NamedTemplates = 2,
}

impl TryFrom<u8> for FfiChatTemplateKind {
    type Error = CoreError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            value if value == Self::None as u8 => Ok(Self::None),
            value if value == Self::Literal as u8 => Ok(Self::Literal),
            value if value == Self::NamedTemplates as u8 => Ok(Self::NamedTemplates),
            _ => Err(CoreError::MismatchedConfig(format!(
                "Unsupported chat template kind: {value}"
            ))),
        }
    }
}

fn optional_string_to_buffer(value: Option<String>) -> (st_owned_buffer_t, bool) {
    match value {
        Some(value) => (string_to_buffer(value), true),
        None => (empty_buffer(), false),
    }
}

#[cfg(test)]
pub(crate) fn empty_runtime_configuration() -> st_runtime_configuration_t {
    st_runtime_configuration_t {
        bos_token: empty_buffer(),
        has_bos_token: false,
        eos_token: empty_buffer(),
        has_eos_token: false,
        unknown_token: empty_buffer(),
        has_unknown_token: false,
        sep_token: empty_buffer(),
        has_sep_token: false,
        pad_token: empty_buffer(),
        has_pad_token: false,
        cls_token: empty_buffer(),
        has_cls_token: false,
        mask_token: empty_buffer(),
        has_mask_token: false,
        additional_special_tokens: ptr::null_mut(),
        additional_special_tokens_len: 0,
        clean_up_tokenization_spaces: true,
        model_max_length: 0,
        has_model_max_length: false,
        chat_template_kind: FfiChatTemplateKind::None as u8,
        chat_template_literal: empty_buffer(),
        named_chat_templates: ptr::null_mut(),
        named_chat_templates_len: 0,
    }
}

#[cfg(test)]
pub(crate) fn empty_tokenizer_descriptor() -> st_tokenizer_descriptor_t {
    st_tokenizer_descriptor_t {
        runtime_configuration: empty_runtime_configuration(),
        bos_token_id: 0,
        has_bos_token_id: false,
        eos_token_id: 0,
        has_eos_token_id: false,
        unknown_token_id: 0,
        has_unknown_token_id: false,
        base_vocab_size: 0,
        total_vocab_size: 0,
    }
}

fn named_chat_template_to_ffi(name: String, template: String) -> st_named_chat_template_t {
    st_named_chat_template_t {
        name: string_to_buffer(name),
        template_text: string_to_buffer(template),
    }
}

fn chat_template_to_ffi(
    value: Option<JsonValue>,
) -> Result<(u8, st_owned_buffer_t, *mut st_named_chat_template_t, usize), CoreError> {
    let Some(value) = value else {
        return Ok((
            FfiChatTemplateKind::None as u8,
            empty_buffer(),
            ptr::null_mut(),
            0,
        ));
    };

    match value {
        JsonValue::String(template) => Ok((
            FfiChatTemplateKind::Literal as u8,
            string_to_buffer(template),
            ptr::null_mut(),
            0,
        )),
        JsonValue::Array(values) => {
            let templates = values
                .into_iter()
                .map(|value| {
                    let JsonValue::Object(mut object) = value else {
                        return Err(CoreError::MismatchedConfig(
                            "Expected named chat templates to be objects".to_owned(),
                        ));
                    };
                    let name = object
                        .remove("name")
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .ok_or_else(|| {
                            CoreError::MismatchedConfig(
                                "Expected named chat template object to contain a string name"
                                    .to_owned(),
                            )
                        })?;
                    let template = object
                        .remove("template")
                        .and_then(|value| value.as_str().map(str::to_owned))
                        .ok_or_else(|| {
                            CoreError::MismatchedConfig(
                                "Expected named chat template object to contain a string template"
                                    .to_owned(),
                            )
                        })?;
                    Ok((name, template))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (templates, templates_len) = vec_to_raw_parts(
                templates
                    .into_iter()
                    .map(|(name, template)| named_chat_template_to_ffi(name, template))
                    .collect(),
            );
            Ok((
                FfiChatTemplateKind::NamedTemplates as u8,
                empty_buffer(),
                templates,
                templates_len,
            ))
        }
        _ => Err(CoreError::MismatchedConfig(
            "Expected chat_template to be a string, an array of named templates, or null"
                .to_owned(),
        )),
    }
}

pub(crate) fn runtime_configuration_to_ffi(
    configuration: RuntimeConfiguration,
) -> Result<st_runtime_configuration_t, CoreError> {
    let RuntimeConfiguration {
        bos_token,
        eos_token,
        unknown_token,
        sep_token,
        pad_token,
        cls_token,
        mask_token,
        additional_special_tokens,
        clean_up_tokenization_spaces,
        model_max_length,
        chat_template,
    } = configuration;

    let (chat_template_kind, chat_template_literal, named_chat_templates, named_chat_templates_len) =
        chat_template_to_ffi(chat_template)?;
    let (bos_token, has_bos_token) = optional_string_to_buffer(bos_token);
    let (eos_token, has_eos_token) = optional_string_to_buffer(eos_token);
    let (unknown_token, has_unknown_token) = optional_string_to_buffer(unknown_token);
    let (sep_token, has_sep_token) = optional_string_to_buffer(sep_token);
    let (pad_token, has_pad_token) = optional_string_to_buffer(pad_token);
    let (cls_token, has_cls_token) = optional_string_to_buffer(cls_token);
    let (mask_token, has_mask_token) = optional_string_to_buffer(mask_token);
    let (additional_special_tokens, additional_special_tokens_len) = vec_to_raw_parts(
        additional_special_tokens
            .into_iter()
            .map(string_to_buffer)
            .collect(),
    );

    Ok(st_runtime_configuration_t {
        bos_token,
        has_bos_token,
        eos_token,
        has_eos_token,
        unknown_token,
        has_unknown_token,
        sep_token,
        has_sep_token,
        pad_token,
        has_pad_token,
        cls_token,
        has_cls_token,
        mask_token,
        has_mask_token,
        additional_special_tokens,
        additional_special_tokens_len,
        clean_up_tokenization_spaces,
        model_max_length: model_max_length.unwrap_or_default(),
        has_model_max_length: model_max_length.is_some(),
        chat_template_kind,
        chat_template_literal,
        named_chat_templates,
        named_chat_templates_len,
    })
}

pub(crate) fn tokenizer_descriptor_to_ffi(
    metadata: TokenizerMetadata,
) -> Result<st_tokenizer_descriptor_t, CoreError> {
    let TokenizerMetadata {
        runtime_configuration,
        bos_token_id,
        eos_token_id,
        unknown_token_id,
        base_vocab_size,
        total_vocab_size,
    } = metadata;

    let (bos_token_id, has_bos_token_id) = optional_i32(bos_token_id);
    let (eos_token_id, has_eos_token_id) = optional_i32(eos_token_id);
    let (unknown_token_id, has_unknown_token_id) = optional_i32(unknown_token_id);

    Ok(st_tokenizer_descriptor_t {
        runtime_configuration: runtime_configuration_to_ffi(runtime_configuration)?,
        bos_token_id,
        has_bos_token_id,
        eos_token_id,
        has_eos_token_id,
        unknown_token_id,
        has_unknown_token_id,
        base_vocab_size,
        total_vocab_size,
    })
}

fn optional_string_from_buffer(
    buffer: &st_owned_buffer_t,
    present: bool,
    field: &str,
) -> Result<Option<String>, CoreError> {
    if present {
        buffer_to_string(buffer, field).map(Some)
    } else {
        Ok(None)
    }
}

fn chat_template_from_ffi(
    configuration: &st_runtime_configuration_t,
) -> Result<Option<JsonValue>, CoreError> {
    match FfiChatTemplateKind::try_from(configuration.chat_template_kind)? {
        FfiChatTemplateKind::None => Ok(None),
        FfiChatTemplateKind::Literal => buffer_to_string(
            &configuration.chat_template_literal,
            "chat_template_literal",
        )
        .map(JsonValue::String)
        .map(Some),
        FfiChatTemplateKind::NamedTemplates => {
            if configuration.named_chat_templates_len == 0 {
                return Ok(Some(JsonValue::Array(Vec::new())));
            }
            if configuration.named_chat_templates.is_null() {
                return Err(CoreError::Internal(
                    "named_chat_templates was null".to_owned(),
                ));
            }

            unsafe {
                slice::from_raw_parts(
                    configuration.named_chat_templates,
                    configuration.named_chat_templates_len,
                )
            }
            .iter()
            .enumerate()
            .map(|(index, template)| {
                let name = buffer_to_string(
                    &template.name,
                    &format!("named_chat_templates[{index}].name"),
                )?;
                let template_text = buffer_to_string(
                    &template.template_text,
                    &format!("named_chat_templates[{index}].template_text"),
                )?;
                Ok(serde_json::json!({
                    "name": name,
                    "template": template_text,
                }))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(JsonValue::Array)
            .map(Some)
        }
    }
}

pub(crate) fn runtime_configuration_from_ffi(
    configuration: &st_runtime_configuration_t,
) -> Result<RuntimeConfiguration, CoreError> {
    Ok(RuntimeConfiguration {
        bos_token: optional_string_from_buffer(
            &configuration.bos_token,
            configuration.has_bos_token,
            "bos_token",
        )?,
        eos_token: optional_string_from_buffer(
            &configuration.eos_token,
            configuration.has_eos_token,
            "eos_token",
        )?,
        unknown_token: optional_string_from_buffer(
            &configuration.unknown_token,
            configuration.has_unknown_token,
            "unknown_token",
        )?,
        sep_token: optional_string_from_buffer(
            &configuration.sep_token,
            configuration.has_sep_token,
            "sep_token",
        )?,
        pad_token: optional_string_from_buffer(
            &configuration.pad_token,
            configuration.has_pad_token,
            "pad_token",
        )?,
        cls_token: optional_string_from_buffer(
            &configuration.cls_token,
            configuration.has_cls_token,
            "cls_token",
        )?,
        mask_token: optional_string_from_buffer(
            &configuration.mask_token,
            configuration.has_mask_token,
            "mask_token",
        )?,
        additional_special_tokens: string_array_from_buffers(
            configuration.additional_special_tokens,
            configuration.additional_special_tokens_len,
            "additional_special_tokens",
        )?,
        clean_up_tokenization_spaces: configuration.clean_up_tokenization_spaces,
        model_max_length: configuration
            .has_model_max_length
            .then_some(configuration.model_max_length),
        chat_template: chat_template_from_ffi(configuration)?,
    })
}
