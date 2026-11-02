use genai::adapter::AdapterKind;
use genai::chat::{ChatMessage, ChatRequest};
use genai::resolver::{AuthData, Endpoint, ServiceTargetResolver};
use genai::{Client, ClientBuilder, ModelIden, ServiceTarget};
use thiserror::Error;

use crate::ai_prompt::{AIPrompt, AIPromptError};
use crate::command::{draft::DraftCommand, explain::ExplainCommand, operate::OperateCommand};
use crate::config::cli::ProviderType;
use crate::config::ProviderInfo;
use crate::error::LumenError;

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
}

enum ProviderBackend {
    GenAI { client: Client, model: String },
}

pub struct LumenProvider {
    backend: ProviderBackend,
    provider_name: String,
}

/// Explicit genai adapter and default endpoint for a provider.
///
/// Every provider maps to its own adapter here. lumen never lets genai infer the
/// adapter from the model name: genai falls back to Ollama for any name it does
/// not know (for example a new DeepSeek model).
fn genai_route(provider: ProviderType, model: &str) -> (AdapterKind, &'static str) {
    match provider {
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
    }
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
    let (adapter_kind, default_endpoint) = genai_route(provider, model);

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

impl LumenProvider {
    pub fn new(
        provider_type: ProviderType,
        api_key: Option<String>,
        model: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self, LumenError> {
        let defaults = ProviderInfo::for_provider(provider_type);
        let model = model.unwrap_or_else(|| defaults.default_model.to_string());

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
        }
    }

    pub async fn explain(&self, command: &ExplainCommand) -> Result<String, ProviderError> {
        let prompt = AIPrompt::build_explain_prompt(command)?;
        self.complete(prompt).await
    }

    pub async fn draft(&self, command: &DraftCommand) -> Result<String, ProviderError> {
        let prompt = AIPrompt::build_draft_prompt(command)?;
        self.complete(prompt).await
    }

    pub async fn operate(&self, command: &OperateCommand) -> Result<String, ProviderError> {
        let prompt = AIPrompt::build_operate_prompt(command.query.as_str())?;
        self.complete(prompt).await
    }

    fn get_model(&self) -> String {
        match &self.backend {
            ProviderBackend::GenAI { model, .. } => model.clone(),
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
            for info in ALL_PROVIDERS {
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
