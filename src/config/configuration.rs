use crate::config::cli::ProviderType;
use crate::config::ProviderInfo;
use crate::error::LumenError;
use dirs::home_dir;
use indoc::indoc;
use serde::{Deserialize, Deserializer};
use serde_json::from_reader;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;

use crate::Cli;

/// Settings resolved from CLI flags, the config file, and environment variables.
#[derive(Debug)]
pub struct LumenConfig {
    pub provider: ProviderType,
    pub model: Option<String>,
    pub base_url: Option<String>,
    pub api_key: Option<String>,
    pub draft: DraftConfig,
    pub theme: Option<String>,
    pub wrap: Option<bool>,
}

/// Per-provider settings under the `providers` object of the config file.
#[derive(Debug, Deserialize, Default, Clone, PartialEq)]
pub struct ProviderSettings {
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

/// The config file as written on disk.
#[derive(Debug, Deserialize)]
pub struct FileConfig {
    #[serde(
        default = "default_ai_provider",
        deserialize_with = "deserialize_ai_provider"
    )]
    pub provider: ProviderType,

    /// Legacy top-level model. `providers.<id>.model` has priority.
    #[serde(default)]
    pub model: Option<String>,

    #[serde(default = "default_api_key")]
    pub api_key: Option<String>,

    #[serde(default)]
    pub providers: HashMap<String, ProviderSettings>,

    #[serde(default = "default_draft_config")]
    pub draft: DraftConfig,

    #[serde(default)]
    pub theme: Option<String>,

    #[serde(default)]
    pub wrap: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
pub struct DraftConfig {
    #[serde(
        default = "default_commit_types",
        deserialize_with = "deserialize_commit_types"
    )]
    pub commit_types: String,
}

fn default_ai_provider() -> ProviderType {
    std::env::var("LUMEN_AI_PROVIDER")
        .unwrap_or_else(|_| "openai".to_string())
        .parse()
        .unwrap_or(ProviderType::Openai)
}

fn deserialize_ai_provider<'de, D>(deserializer: D) -> Result<ProviderType, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse().map_err(serde::de::Error::custom)
}

fn default_commit_types() -> String {
    indoc! {r#"
    {
        "docs": "Documentation only changes",
        "style": "Changes that do not affect the meaning of the code",
        "refactor": "A code change that neither fixes a bug nor adds a feature",
        "perf": "A code change that improves performance",
        "test": "Adding missing tests or correcting existing tests",
        "build": "Changes that affect the build system or external dependencies",
        "ci": "Changes to our CI configuration files and scripts",
        "chore": "Other changes that don't modify src or test files",
        "revert": "Reverts a previous commit",
        "feat": "A new feature",
        "fix": "A bug fix"
    }
    "#}
    .to_string()
}

fn default_api_key() -> Option<String> {
    std::env::var("LUMEN_API_KEY").ok()
}

fn deserialize_commit_types<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let commit_types_map: HashMap<String, String> = HashMap::deserialize(deserializer)?;
    serde_json::to_string(&commit_types_map).map_err(serde::de::Error::custom)
}

fn default_draft_config() -> DraftConfig {
    DraftConfig {
        commit_types: default_commit_types(),
    }
}

fn default_config_path() -> Option<String> {
    home_dir().and_then(|mut path| {
        path.push(".config/lumen/lumen.config.json");
        path.exists()
            .then_some(path)
            .and_then(|p| p.to_str().map(|s| s.to_string()))
    })
}

/// Model precedence: `-m`, then `providers.<active>.model`, then the legacy
/// top-level `model`, then `LUMEN_AI_MODEL`. `None` means the provider default.
fn resolve_model(
    cli_model: Option<&str>,
    file: &FileConfig,
    provider: ProviderType,
    env_model: Option<String>,
) -> Option<String> {
    let provider_id = ProviderInfo::for_provider(provider).id;
    cli_model
        .map(str::to_string)
        .or_else(|| {
            file.providers
                .get(provider_id)
                .and_then(|p| p.model.clone())
        })
        .or_else(|| file.model.clone())
        .or(env_model)
}

impl LumenConfig {
    pub fn build(cli: &Cli) -> Result<Self, LumenError> {
        let file = if let Some(config_path) = &cli.config {
            FileConfig::from_file(config_path)?
        } else {
            match default_config_path() {
                Some(path) => FileConfig::from_file(&path)?,
                None => FileConfig::default(),
            }
        };
        Ok(Self::resolve(
            cli,
            file,
            std::env::var("LUMEN_AI_MODEL").ok(),
        ))
    }

    fn resolve(cli: &Cli, file: FileConfig, env_model: Option<String>) -> Self {
        let provider = cli.provider.unwrap_or(file.provider);
        let model = resolve_model(cli.model.as_deref(), &file, provider, env_model);
        let base_url = file
            .providers
            .get(ProviderInfo::for_provider(provider).id)
            .and_then(|p| p.base_url.clone());
        let api_key = cli.api_key.clone().or(file.api_key);

        LumenConfig {
            provider,
            model,
            base_url,
            api_key,
            draft: file.draft,
            theme: file.theme,
            wrap: file.wrap,
        }
    }
}

impl FileConfig {
    pub fn from_file(file_path: &str) -> Result<Self, LumenError> {
        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        from_reader(reader).map_err(|e| LumenError::InvalidConfiguration(e.to_string()))
    }
}

impl Default for FileConfig {
    fn default() -> Self {
        FileConfig {
            provider: default_ai_provider(),
            model: None,
            api_key: default_api_key(),
            providers: HashMap::new(),
            draft: default_draft_config(),
            theme: None,
            wrap: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn file(json: &str) -> FileConfig {
        serde_json::from_str(json).unwrap()
    }

    fn cli(args: &[&str]) -> Cli {
        let mut all = vec!["lumen"];
        all.extend_from_slice(args);
        all.push("draft");
        Cli::try_parse_from(all).unwrap()
    }

    #[test]
    fn cli_model_wins_over_everything() {
        let f = file(
            r#"{"provider":"deepseek","model":"top","providers":{"deepseek":{"model":"per"}}}"#,
        );
        let c = LumenConfig::resolve(&cli(&["-m", "flag"]), f, Some("env".into()));
        assert_eq!(c.model.as_deref(), Some("flag"));
    }

    #[test]
    fn provider_model_wins_over_top_level_and_env() {
        let f = file(
            r#"{"provider":"deepseek","model":"top","providers":{"deepseek":{"model":"per"}}}"#,
        );
        let c = LumenConfig::resolve(&cli(&[]), f, Some("env".into()));
        assert_eq!(c.model.as_deref(), Some("per"));
    }

    #[test]
    fn top_level_model_wins_over_env() {
        let f = file(r#"{"provider":"deepseek","model":"top"}"#);
        let c = LumenConfig::resolve(&cli(&[]), f, Some("env".into()));
        assert_eq!(c.model.as_deref(), Some("top"));
    }

    #[test]
    fn env_model_used_last() {
        let f = file(r#"{"provider":"deepseek"}"#);
        let c = LumenConfig::resolve(&cli(&[]), f, Some("env".into()));
        assert_eq!(c.model.as_deref(), Some("env"));
    }

    #[test]
    fn model_of_other_provider_is_not_used() {
        let f = file(r#"{"provider":"claude","providers":{"deepseek":{"model":"deepseek-v4"}}}"#);
        let c = LumenConfig::resolve(&cli(&[]), f, None);
        assert_eq!(c.model, None);
    }

    #[test]
    fn provider_flag_selects_that_providers_settings() {
        let f = file(
            r#"{"provider":"claude","providers":{"ollama":{"model":"qwen3","base_url":"http://box:11434/v1/"}}}"#,
        );
        let c = LumenConfig::resolve(&cli(&["-p", "ollama"]), f, None);
        assert_eq!(c.provider, ProviderType::Ollama);
        assert_eq!(c.model.as_deref(), Some("qwen3"));
        assert_eq!(c.base_url.as_deref(), Some("http://box:11434/v1/"));
    }
}
