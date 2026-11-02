use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatRequest};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ClientBuilder, ModelIden, ServiceTarget};
use thiserror::Error;

mod apple_fm;

use crate::ai_prompt::{AIPrompt, AIPromptError, PromptDiff};
use crate::command::{draft::DraftCommand, explain::ExplainCommand, operate::OperateCommand};
use crate::config::cli::ProviderType;
use crate::config::ProviderInfo;
use crate::error::LumenError;
use crate::prompt_budget::{estimate_tokens, fit_diff, CHARS_PER_TOKEN};

#[derive(Error, Debug)]
pub enum ProviderError {
    #[error("AI request failed: {0}")]
    GenAIError(#[from] genai::Error),

    #[error("API request failed: {0}")]
    RequestError(#[from] reqwest::Error),

    #[error("No completion content in response")]
    NoCompletionChoice,

    #[error(transparent)]
    AIPromptError(#[from] AIPromptError),

    #[error("Apple Foundation Models CLI '{0}' not found. It ships as /usr/bin/fm on supported macOS versions.")]
    AppleFmNotFound(String),

    #[error("Apple Foundation Models: {0}")]
    AppleContextExceeded(String),

    #[error("Apple Foundation Models request failed: {0}")]
    AppleFmFailed(String),

    #[error("prompt is too large: about {estimated} tokens, but the budget is {budget} tokens, even with only the list of changed files. Make the change smaller, or raise `max_input_tokens` under `providers.<provider>` in the lumen config.")]
    PromptOverBudget { estimated: usize, budget: usize },
}

enum ProviderBackend {
    GenAI { client: Client, model: String },
    AppleFm { program: String, model: String },
}

pub struct LumenProvider {
    backend: ProviderBackend,
    provider_name: String,
    /// Input-token budget for prompts that contain a diff.
    max_input_tokens: usize,
}

/// How a provider is reached.
#[derive(Debug, PartialEq)]
enum Route {
    /// genai with an explicit adapter and default endpoint.
    GenAI(AdapterKind, &'static str),
    /// The local `fm` CLI.
    AppleFm,
}

/// Explicit route for a provider.
///
/// Every provider maps to its own adapter here. lumen never lets genai infer the
/// adapter from the model name: genai falls back to Ollama for any name it does
/// not know (for example a new DeepSeek model).
fn route(provider: ProviderType, model: &str) -> Route {
    let (kind, endpoint) = match provider {
        ProviderType::Apple => return Route::AppleFm,
        // OpenAI serves some models (codex, pro) only via the Responses API.
        ProviderType::Openai => match AdapterKind::from_model(model) {
            Ok(AdapterKind::OpenAIResp) => (AdapterKind::OpenAIResp, "https://api.openai.com/v1/"),
            _ => (AdapterKind::OpenAI, "https://api.openai.com/v1/"),
        },
        ProviderType::Claude => (AdapterKind::Anthropic, "https://api.anthropic.com/v1/"),
        ProviderType::Gemini => (
            AdapterKind::Gemini,
            "https://generativelanguage.googleapis.com/v1beta/",
        ),
        ProviderType::Xai => (AdapterKind::Xai, "https://api.x.ai/v1/"),
        ProviderType::Deepseek => (AdapterKind::DeepSeek, "https://api.deepseek.com/v1/"),
        ProviderType::Ollama => (AdapterKind::Ollama, "http://localhost:11434/v1/"),
        ProviderType::Groq => (AdapterKind::OpenAI, "https://api.groq.com/openai/v1/"),
        ProviderType::OpencodeZen => (AdapterKind::OpenAI, "https://opencode.ai/zen/v1/"),
        ProviderType::Openrouter => (AdapterKind::OpenAI, "https://openrouter.ai/api/v1/"),
        // Trailing slash is required for URL joining to work correctly
        ProviderType::Vercel => (AdapterKind::OpenAI, "https://ai-gateway.vercel.sh/v1/"),
    };
    Route::GenAI(kind, endpoint)
}

/// Builds the service target for a request. The endpoint and adapter come only
/// from the provider (and the optional `base_url` override), never from the model name.
fn service_target(
    provider: ProviderType,
    model: &str,
    base_url: Option<&str>,
    api_key: Option<&str>,
) -> ServiceTarget {
    let defaults = ProviderInfo::for_provider(provider);
    let Route::GenAI(adapter_kind, default_endpoint) = route(provider, model) else {
        unreachable!("service_target is only built for genai providers")
    };

    let endpoint = match base_url {
        Some(url) if url.ends_with('/') => Endpoint::from_owned(url.to_string()),
        Some(url) => Endpoint::from_owned(format!("{url}/")),
        None => Endpoint::from_static(default_endpoint),
    };

    let auth = match api_key {
        Some(key) => AuthData::from_single(key.to_string()),
        None if defaults.env_key.is_empty() => AuthData::from_single(defaults.id.to_string()),
        None => AuthData::from_env(defaults.env_key),
    };

    ServiceTarget {
        endpoint,
        auth,
        model: ModelIden::new(adapter_kind, model.to_string()),
    }
}

/// Build a prompt that fits the input-token budget.
///
/// The full diff is used when it fits. Otherwise the largest file bodies are
/// replaced by summary lines, a note goes to stderr, and the prompt names the
/// omitted files. Fails when even the list of changed files does not fit.
fn fit_prompt(
    max_input_tokens: usize,
    diff: &str,
    build: impl Fn(&PromptDiff) -> Result<AIPrompt, AIPromptError>,
) -> Result<AIPrompt, ProviderError> {
    let budget_chars = max_input_tokens.saturating_mul(CHARS_PER_TOKEN);
    let full = build(&PromptDiff::full(diff))?;
    let full_chars = full.chars();
    if full_chars <= budget_chars {
        return Ok(full);
    }

    let over_budget = || ProviderError::PromptOverBudget {
        estimated: estimate_tokens(full_chars),
        budget: max_input_tokens,
    };
    let overhead = full_chars.saturating_sub(diff.chars().count());
    let available = budget_chars.checked_sub(overhead).ok_or_else(over_budget)?;
    let fitted = fit_diff(diff, available).ok_or_else(over_budget)?;

    let prompt = build(&PromptDiff {
        text: &fitted.text,
        omitted: &fitted.omitted,
    })?;
    if prompt.chars() > budget_chars {
        return Err(over_budget());
    }
    eprintln!(
        "note: diff content of {} file(s) omitted (prompt over the {max_input_tokens}-token budget)",
        fitted.omitted.len()
    );
    Ok(prompt)
}

impl LumenProvider {
    pub fn new(
        provider_type: ProviderType,
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
        max_input_tokens: usize,
    ) -> Result<Self, LumenError> {
        let defaults = ProviderInfo::for_provider(provider_type);
        let model = model.unwrap_or_else(|| defaults.default_model.to_string());

        if route(provider_type, &model) == Route::AppleFm {
            return Ok(Self {
                backend: ProviderBackend::AppleFm {
                    program: "fm".to_string(),
                    model,
                },
                provider_name: defaults.display_name.to_string(),
                max_input_tokens,
            });
        }

        let target_resolver = ServiceTargetResolver::from_resolver_fn(
            move |target: ServiceTarget| -> Result<ServiceTarget, genai::resolver::Error> {
                Ok(service_target(
                    provider_type,
                    &target.model.model_name,
                    base_url.as_deref(),
                    api_key.as_deref(),
                ))
            },
        );

        let client = ClientBuilder::default()
            .with_service_target_resolver(target_resolver)
            .build();

        Ok(Self {
            backend: ProviderBackend::GenAI { client, model },
            provider_name: defaults.display_name.to_string(),
            max_input_tokens,
        })
    }

    async fn complete(&self, prompt: AIPrompt) -> Result<String, ProviderError> {
        match &self.backend {
            ProviderBackend::GenAI { client, model } => {
                let chat_req = ChatRequest::new(vec![
                    ChatMessage::system(prompt.system_prompt),
                    ChatMessage::user(prompt.user_prompt),
                ]);

                let response = client.exec_chat(model, chat_req, None).await?;

                response
                    .first_text()
                    .map(|s| s.to_string())
                    .ok_or(ProviderError::NoCompletionChoice)
            }
            ProviderBackend::AppleFm { program, model } => {
                apple_fm::complete(program, model, &prompt.system_prompt, &prompt.user_prompt).await
            }
        }
    }

    pub async fn explain(&self, command: &ExplainCommand) -> Result<String, ProviderError> {
        let prompt = fit_prompt(
            self.max_input_tokens,
            command.git_entity.diff_text(),
            |diff| AIPrompt::build_explain_prompt(command, diff),
        )?;
        self.complete(prompt).await
    }

    pub async fn draft(&self, command: &DraftCommand) -> Result<String, ProviderError> {
        let prompt = fit_prompt(
            self.max_input_tokens,
            command.git_entity.diff_text(),
            |diff| AIPrompt::build_draft_prompt(command, diff),
        )?;
        self.complete(prompt).await
    }

    pub async fn operate(&self, command: &OperateCommand) -> Result<String, ProviderError> {
        let prompt = AIPrompt::build_operate_prompt(command.query.as_str())?;
        self.complete(prompt).await
    }

    fn get_model(&self) -> String {
        match &self.backend {
            ProviderBackend::GenAI { model, .. } | ProviderBackend::AppleFm { model, .. } => {
                model.clone()
            }
        }
    }
}

impl std::fmt::Display for LumenProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.provider_name, self.get_model())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ALL_PROVIDERS;

    /// Test prompt builder: a fixed system prompt plus the diff and the omitted-files note.
    fn echo_prompt(diff: &PromptDiff) -> Result<AIPrompt, AIPromptError> {
        Ok(AIPrompt {
            system_prompt: "system".to_string(),
            user_prompt: format!("{}|omitted={}", diff.text, diff.omitted.join(",")),
        })
    }

    fn file(path: &str, lines: usize) -> String {
        let mut s =
            format!("diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1 @@\n");
        for i in 0..lines {
            s.push_str(&format!("+{i} {path}\n"));
        }
        s
    }

    #[test]
    fn prompt_within_budget_keeps_full_diff() {
        let diff = file("a.rs", 10);
        let p = fit_prompt(10_000, &diff, echo_prompt).unwrap();
        assert_eq!(p.user_prompt, format!("{diff}|omitted="));
    }

    #[test]
    fn prompt_over_budget_omits_largest_file_and_names_it() {
        let diff = format!("{}{}", file("small.rs", 3), file("huge.bin.js", 20_000));
        let p = fit_prompt(500, &diff, echo_prompt).unwrap();
        assert!(p.chars() <= 500 * CHARS_PER_TOKEN);
        assert!(
            p.user_prompt.contains("+2 small.rs"),
            "small file keeps content"
        );
        assert!(p
            .user_prompt
            .contains("huge.bin.js: +20000 -0 lines (diff omitted)"));
        assert!(p.user_prompt.ends_with("|omitted=huge.bin.js"));
    }

    #[test]
    fn prompt_fails_when_file_list_does_not_fit() {
        let diff: String = (0..200).map(|i| file(&format!("f{i}.rs"), 5)).collect();
        let err = fit_prompt(100, &diff, echo_prompt).unwrap_err();
        let ProviderError::PromptOverBudget { estimated, budget } = err else {
            panic!("expected PromptOverBudget, got {err:?}");
        };
        assert_eq!(budget, 100);
        assert!(estimated > budget);
    }

    fn expected_adapter(provider: ProviderType) -> AdapterKind {
        match provider {
            ProviderType::Openai
            | ProviderType::Groq
            | ProviderType::OpencodeZen
            | ProviderType::Openrouter
            | ProviderType::Vercel => AdapterKind::OpenAI,
            ProviderType::Claude => AdapterKind::Anthropic,
            ProviderType::Gemini => AdapterKind::Gemini,
            ProviderType::Xai => AdapterKind::Xai,
            ProviderType::Deepseek => AdapterKind::DeepSeek,
            ProviderType::Ollama => AdapterKind::Ollama,
            ProviderType::Apple => unreachable!("not a genai provider"),
        }
    }

    #[test]
    fn deepseek_model_unknown_to_genai_routes_to_deepseek() {
        let t = service_target(ProviderType::Deepseek, "deepseek-v4-pro", None, Some("k"));
        assert_eq!(t.model.adapter_kind, AdapterKind::DeepSeek);
        assert_eq!(t.endpoint.base_url(), "https://api.deepseek.com/v1/");
    }

    #[test]
    fn model_name_never_changes_the_provider_route() {
        // Names that genai would route elsewhere (Ollama fallback, Gemini, Anthropic, ...).
        for name in [
            "my-model",
            "gemini-2.5-flash",
            "claude-x",
            "deepseek-chat",
            "grok-4",
        ] {
            for info in ALL_PROVIDERS
                .iter()
                .filter(|p| p.provider_type != ProviderType::Apple)
            {
                let t = service_target(info.provider_type, name, None, None);
                assert_eq!(
                    t.model.adapter_kind,
                    expected_adapter(info.provider_type),
                    "provider {} with model {name}",
                    info.id
                );
            }
        }
    }

    #[test]
    fn apple_routes_to_fm_for_any_model_name() {
        for name in ["system", "pcc", "gpt-5"] {
            assert_eq!(route(ProviderType::Apple, name), Route::AppleFm);
        }
    }

    #[test]
    fn apple_provider_defaults_to_on_device_model() {
        let p = LumenProvider::new(ProviderType::Apple, None, None, None, 1_000).unwrap();
        assert_eq!(p.to_string(), "Apple Foundation Models (system)");
        let p =
            LumenProvider::new(ProviderType::Apple, None, Some("pcc".into()), None, 1_000).unwrap();
        assert_eq!(p.to_string(), "Apple Foundation Models (pcc)");
    }

    #[test]
    fn openai_responses_models_keep_the_responses_adapter() {
        let t = service_target(ProviderType::Openai, "gpt-5-codex", None, None);
        assert_eq!(t.model.adapter_kind, AdapterKind::OpenAIResp);
    }

    #[test]
    fn base_url_override_replaces_endpoint_and_gets_trailing_slash() {
        let t = service_target(
            ProviderType::Ollama,
            "llama3.2",
            Some("http://box:11434/v1"),
            None,
        );
        assert_eq!(t.endpoint.base_url(), "http://box:11434/v1/");
    }

    #[test]
    fn explicit_key_is_used_otherwise_provider_env() {
        let t = service_target(ProviderType::Deepseek, "deepseek-chat", None, Some("sk-1"));
        assert!(matches!(t.auth, AuthData::Key(ref k) if k == "sk-1"));
        let t = service_target(ProviderType::Deepseek, "deepseek-chat", None, None);
        assert!(matches!(t.auth, AuthData::FromEnv(ref e) if e == "DEEPSEEK_API_KEY"));
    }
}
