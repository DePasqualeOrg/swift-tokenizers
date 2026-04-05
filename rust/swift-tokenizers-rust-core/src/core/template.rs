use minijinja::Environment;
use serde_json::Value as JsonValue;

use crate::core::error::CoreError;

pub(crate) fn make_environment() -> Environment<'static> {
    let mut environment = Environment::new();
    environment.add_filter("startswith", |value: String, prefix: String| {
        value.starts_with(&prefix)
    });
    environment.add_filter("endswith", |value: String, suffix: String| {
        value.ends_with(&suffix)
    });
    environment.set_keep_trailing_newline(false);
    environment.set_trim_blocks(true);
    environment.set_lstrip_blocks(true);
    environment
}

pub(crate) fn render(
    environment: &Environment<'static>,
    template: &str,
    context: &JsonValue,
) -> Result<String, CoreError> {
    environment
        .render_str(&normalize_template_source(template), context)
        .map_err(|err| CoreError::ChatTemplate(err.to_string()))
}

fn normalize_template_source(template: &str) -> String {
    template
        .replace(".startswith(", "|startswith(")
        .replace(".endswith(", "|endswith(")
}
