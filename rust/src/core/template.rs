// Copyright © Anthony DePasquale

use std::sync::LazyLock;

use chrono::Local;
use minijinja::Environment;
use minijinja_contrib::pycompat::unknown_method_callback;
use regex::Regex;
use serde_json::Value as JsonValue;

use crate::core::error::CoreError;

pub(crate) fn make_environment() -> Environment<'static> {
    let mut environment = Environment::new();
    // Python-Jinja methods (`dict.get`, `str.strip`, `list.count`, `str.startswith`,
    // `str.endswith`, etc.) that chat templates routinely use but minijinja does not
    // ship natively. Enabling `pycompat::unknown_method_callback` is the minijinja
    // maintainer's recommended path for Python compatibility (mitsuhiko/minijinja#815).
    environment.set_unknown_method_callback(unknown_method_callback);
    // `strftime_now(fmt)` is an extension exposed by Python transformers' Jinja
    // environment — not a standard Jinja function. Real-world chat templates
    // (e.g. smollm3, granite) call it to stamp the current date into system prompts,
    // so omitting it surfaces as `UnknownFunction: strftime_now is unknown` at render
    // time. Format tokens follow Python strftime; chrono's format syntax matches on
    // the tokens these templates actually use (`%Y`, `%B`, `%d`, `%H`, `%M`, `%S`).
    // Local time matches Python's default `datetime.now()`.
    environment.add_function("strftime_now", |format: String| -> String {
        Local::now().format(&format).to_string()
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
    let template_source = normalize_template_source(template);
    let template = environment
        .template_from_str(&template_source)
        .map_err(|err| {
            CoreError::ChatTemplate(format!("Failed to compile chat template: {err}"))
        })?;

    template
        .render(context)
        .map_err(|err| CoreError::ChatTemplate(format!("Failed to render chat template: {err}")))
}

// Matches `{% generation %}` / `{% endgeneration %}` in any whitespace-control variant.
//
// These blocks are a chat-template extension layered on top of Python Jinja2 — they
// are not part of standard Jinja syntax, so minijinja does not (and, per the minijinja
// maintainer's reply in mitsuhiko/minijinja#815, will not) parse them as first-class
// tags. Python transformers uses the blocks to record assistant-generated spans so
// callers can retrieve an `assistant_mask` array alongside the tokenized output.
//
// We do not compute that mask, so stripping the tags here is semantically equivalent to
// how swift-jinja's interpreter handles them: parse the block as first-class, then
// render its body with no other effect. Stripping at the source level lets minijinja
// treat the body as ordinary template content.
static GENERATION_TAG: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\{%-?\s*(?:end)?generation\s*-?%\}").expect("valid regex"));

fn normalize_template_source(template: &str) -> String {
    GENERATION_TAG.replace_all(template, "").into_owned()
}
