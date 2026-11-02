use crate::config::cli::ProviderType;
use crate::config::secrets::{OsKeychain, SecretStore};
use crate::config::{ProviderInfo, ALL_PROVIDERS};
use crate::error::LumenError;
use dirs::home_dir;
use inquire::{Confirm, Password, PasswordDisplayMode, Select, Text};
use serde_json::{json, Value};
use std::fmt;
use std::fs;

/// Wrapper for display in the selection prompt
struct ProviderChoice(&'static ProviderInfo);

impl fmt::Display for ProviderChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.display_name)
    }
}

/// Command to handle interactive configuration of Lumen features.
pub struct ConfigureCommand;

fn prompt_err(e: inquire::InquireError) -> LumenError {
    LumenError::ConfigurationError(e.to_string())
}

fn secret_store_err(e: String) -> LumenError {
    LumenError::ConfigurationError(format!("secret store: {e}"))
}

/// The provider that the config file selects now (before this run changes it).
fn active_provider(config: &Value) -> ProviderType {
    config
        .get("provider")
        .and_then(Value::as_str)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| {
            std::env::var("LUMEN_AI_PROVIDER")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(ProviderType::Openai)
        })
}

/// Removes the plaintext `api_key` and returns it with the provider that owns it.
/// Returns `None` if there is no key, or its owner needs no key.
fn plaintext_key(config: &Value) -> Option<(&'static ProviderInfo, String)> {
    let key = config.get("api_key")?.as_str()?.to_string();
    let owner = ProviderInfo::for_provider(active_provider(config));
    owner.requires_key().then_some((owner, key))
}

/// Applies the provider and model choice to the config JSON.
///
/// A legacy top-level `model` moves to `providers.<previously active>.model`
/// (unless that is already set). The model of the chosen provider is written to
/// `providers.<id>.model`, or removed when `model` is `None`.
fn apply_selection(config: &mut Value, provider: &ProviderInfo, model: Option<&str>) {
    if !config.is_object() {
        *config = json!({});
    }
    let previous = ProviderInfo::for_provider(active_provider(config)).id;
    let obj = config.as_object_mut().expect("config is an object");

    if !obj.get("providers").is_some_and(Value::is_object) {
        obj.insert("providers".into(), json!({}));
    }
    if let Some(legacy) = obj.remove("model") {
        let providers = obj["providers"].as_object_mut().expect("object");
        let entry = providers.entry(previous).or_insert_with(|| json!({}));
        if entry.get("model").is_none() {
            entry["model"] = legacy;
        }
    }

    obj.insert("provider".into(), json!(provider.id));

    let providers = obj["providers"].as_object_mut().expect("object");
    let entry = providers.entry(provider.id).or_insert_with(|| json!({}));
    match model {
        Some(m) => entry["model"] = json!(m),
        None => {
            entry.as_object_mut().map(|e| e.remove("model"));
        }
    }
    if entry.as_object().is_some_and(|e| e.is_empty()) {
        providers.remove(provider.id);
    }
}

impl ConfigureCommand {
    /// Executes the interactive configuration wizard.
    ///
    /// 1. Offers to move a plaintext `api_key` from the config file to the secure store
    /// 2. Prompts for the AI provider
    /// 3. Asks for the API key (masked) and stores it in the secure store
    /// 4. Asks for a custom model name
    /// 5. Saves `~/.config/lumen/lumen.config.json`
    pub fn execute() -> Result<(), LumenError> {
        Self::run(&OsKeychain)
    }

    fn run(store: &dyn SecretStore) -> Result<(), LumenError> {
        println!("\n  \x1b[1;36mLumen Configuration\x1b[0m\n");

        let config_file = Self::get_config_path()?.join("lumen.config.json");
        let mut config: Value = if config_file.exists() {
            let content = fs::read_to_string(&config_file)?;
            serde_json::from_str(&content).unwrap_or_else(|_| json!({}))
        } else {
            json!({})
        };

        Self::offer_migration(&mut config, store)?;

        let provider = Self::select_provider()?;
        Self::configure_api_key(provider, store)?;
        let model = Self::get_model_name(provider)?;
        apply_selection(&mut config, provider, model.as_deref());

        fs::create_dir_all(config_file.parent().expect("config dir"))?;
        fs::write(&config_file, serde_json::to_string_pretty(&config)?)?;

        println!(
            "\n  \x1b[1;32m✓\x1b[0m Configuration saved to \x1b[2m{}\x1b[0m\n",
            config_file.display()
        );
        Ok(())
    }

    /// Offers to move the plaintext `api_key` to the secure store. The key is
    /// removed from the file only after the store accepted it.
    fn offer_migration(config: &mut Value, store: &dyn SecretStore) -> Result<(), LumenError> {
        let Some((owner, key)) = plaintext_key(config) else {
            return Ok(());
        };
        let question = format!(
            "The config file has a plaintext api_key for {}. Move it to the secure store and delete it from the file?",
            owner.display_name
        );
        if Confirm::new(&question)
            .with_default(true)
            .prompt()
            .map_err(prompt_err)?
        {
            store.set(owner.id, &key).map_err(secret_store_err)?;
            config.as_object_mut().map(|o| o.remove("api_key"));
            println!("  \x1b[2mMoved the key to the secure store.\x1b[0m");
        }
        Ok(())
    }

    /// Prompts the user to select an AI provider from the supported list.
    fn select_provider() -> Result<&'static ProviderInfo, LumenError> {
        let options: Vec<ProviderChoice> = ALL_PROVIDERS.iter().map(ProviderChoice).collect();

        let selection = Select::new("Select your default AI provider:", options)
            .with_help_message("↑↓ to move, enter to select, type to filter")
            .prompt()
            .map_err(prompt_err)?;

        Ok(selection.0)
    }

    /// Reads the API key with masked input and saves it in the secure store.
    /// Empty input keeps the stored key; the user can then remove it.
    fn configure_api_key(
        provider: &ProviderInfo,
        store: &dyn SecretStore,
    ) -> Result<(), LumenError> {
        if !provider.requires_key() {
            println!(
                "\n  \x1b[2m{} runs locally — no API key needed.\x1b[0m",
                provider.display_name
            );
            return Ok(());
        }

        let stored = match store.get(provider.id) {
            Ok(v) => v.is_some(),
            Err(e) => {
                eprintln!("  \x1b[93mwarning:\x1b[0m could not read the secure store: {e}");
                false
            }
        };
        let help = if stored {
            "A key is stored. Leave empty to keep it.".to_string()
        } else {
            format!("Leave empty to use {} instead.", provider.env_key)
        };

        let key = Password::new("Enter your API key:")
            .without_confirmation()
            .with_display_mode(PasswordDisplayMode::Masked)
            .with_help_message(&help)
            .prompt()
            .map_err(prompt_err)?;

        if !key.trim().is_empty() {
            store
                .set(provider.id, key.trim())
                .map_err(secret_store_err)?;
            println!("  \x1b[2mKey saved in the secure store.\x1b[0m");
        } else if stored
            && Confirm::new("Remove the stored key?")
                .with_default(false)
                .prompt()
                .map_err(prompt_err)?
        {
            store.delete(provider.id).map_err(secret_store_err)?;
            println!("  \x1b[2mStored key removed.\x1b[0m");
        }
        Ok(())
    }

    /// Prompts the user for a custom model name.
    /// Returns `None` if the user accepts the default model by pressing Enter.
    fn get_model_name(provider: &ProviderInfo) -> Result<Option<String>, LumenError> {
        let prompt = format!(
            "Enter model name (leave empty for default: {}):",
            provider.default_model
        );

        let model = Text::new(&prompt)
            .with_help_message("Press Enter to use the default model")
            .prompt()
            .map_err(prompt_err)?;

        Ok((!model.is_empty()).then_some(model))
    }

    /// Resolves the path to the configuration directory (`~/.config/lumen`).
    fn get_config_path() -> Result<std::path::PathBuf, LumenError> {
        let mut path = home_dir().ok_or_else(|| {
            LumenError::ConfigurationError("Could not determine home directory".to_string())
        })?;
        path.push(".config");
        path.push("lumen");
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(p: ProviderType) -> &'static ProviderInfo {
        ProviderInfo::for_provider(p)
    }

    #[test]
    fn writes_provider_and_per_provider_model() {
        let mut c = json!({});
        apply_selection(&mut c, info(ProviderType::Deepseek), Some("deepseek-v4"));
        assert_eq!(
            c,
            json!({"provider":"deepseek","providers":{"deepseek":{"model":"deepseek-v4"}}})
        );
    }

    #[test]
    fn legacy_model_moves_to_previously_active_provider() {
        let mut c = json!({"provider":"claude","model":"claude-opus","theme":"nord"});
        apply_selection(&mut c, info(ProviderType::Deepseek), None);
        assert_eq!(
            c,
            json!({"provider":"deepseek","theme":"nord","providers":{"claude":{"model":"claude-opus"}}})
        );
    }

    #[test]
    fn legacy_model_does_not_overwrite_existing_provider_model() {
        let mut c =
            json!({"provider":"claude","model":"old","providers":{"claude":{"model":"new"}}});
        apply_selection(&mut c, info(ProviderType::Claude), Some("new"));
        assert_eq!(c["providers"]["claude"]["model"], json!("new"));
        assert!(c.get("model").is_none());
    }

    #[test]
    fn empty_model_removes_only_model_and_keeps_base_url() {
        let mut c = json!({"provider":"ollama","providers":{"ollama":{"model":"x","base_url":"http://b/"}}});
        apply_selection(&mut c, info(ProviderType::Ollama), None);
        assert_eq!(c["providers"]["ollama"], json!({"base_url":"http://b/"}));
    }

    #[test]
    fn plaintext_key_belongs_to_the_active_provider() {
        let c = json!({"provider":"deepseek","api_key":"sk-1"});
        let (owner, key) = plaintext_key(&c).unwrap();
        assert_eq!((owner.id, key.as_str()), ("deepseek", "sk-1"));
    }

    #[test]
    fn plaintext_key_of_keyless_provider_is_not_migrated() {
        assert!(plaintext_key(&json!({"provider":"ollama","api_key":"x"})).is_none());
        assert!(plaintext_key(&json!({"provider":"claude"})).is_none());
    }
}
