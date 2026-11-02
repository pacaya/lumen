//! API key storage in the OS secret store and key resolution.
use crate::config::ProviderInfo;
use crate::error::LumenError;

/// Keychain service name. The account is the provider id (for example `deepseek`).
pub const KEYCHAIN_SERVICE: &str = "lumen";

/// Port to the OS secret store. One item per provider id.
pub trait SecretStore {
    /// `Ok(None)` when no item exists.
    fn get(&self, provider_id: &str) -> Result<Option<String>, String>;
    fn set(&self, provider_id: &str, secret: &str) -> Result<(), String>;
    /// `Ok(false)` when no item existed.
    fn delete(&self, provider_id: &str) -> Result<bool, String>;
}

/// The platform secret store: macOS Keychain, Windows Credential Manager,
/// or Secret Service on Linux.
pub struct OsKeychain;

impl OsKeychain {
    fn entry(provider_id: &str) -> Result<keyring::Entry, String> {
        keyring::Entry::new(KEYCHAIN_SERVICE, provider_id).map_err(|e| e.to_string())
    }
}

impl SecretStore for OsKeychain {
    fn get(&self, provider_id: &str) -> Result<Option<String>, String> {
        match Self::entry(provider_id)?.get_password() {
            Ok(secret) => Ok(Some(secret)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(e.to_string()),
        }
    }

    fn set(&self, provider_id: &str, secret: &str) -> Result<(), String> {
        Self::entry(provider_id)?
            .set_password(secret)
            .map_err(|e| e.to_string())
    }

    fn delete(&self, provider_id: &str) -> Result<bool, String> {
        match Self::entry(provider_id)?.delete_credential() {
            Ok(()) => Ok(true),
            Err(keyring::Error::NoEntry) => Ok(false),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Resolves the API key for `provider`.
///
/// Precedence: `-k` flag, `LUMEN_API_KEY`, the provider env var, the secret
/// store item, then the deprecated plaintext `api_key` of the config file.
/// Returns `Ok(None)` for providers that need no key. A secret store failure
/// other than "not found" produces a warning and resolution continues.
pub fn resolve_api_key(
    provider: &ProviderInfo,
    flag_key: Option<&str>,
    file_key: Option<&str>,
    env: &dyn Fn(&str) -> Option<String>,
    store: &dyn SecretStore,
    warn: &mut dyn FnMut(String),
) -> Result<Option<String>, LumenError> {
    if !provider.requires_key() {
        return Ok(None);
    }

    let non_empty = |v: Option<String>| v.filter(|s| !s.trim().is_empty());

    if let Some(key) = non_empty(flag_key.map(str::to_string)) {
        return Ok(Some(key));
    }
    if let Some(key) = non_empty(env("LUMEN_API_KEY")) {
        return Ok(Some(key));
    }
    if let Some(key) = non_empty(env(provider.env_key)) {
        return Ok(Some(key));
    }
    match store.get(provider.id) {
        Ok(found) => {
            if let Some(key) = non_empty(found) {
                return Ok(Some(key));
            }
        }
        Err(e) => warn(format!(
            "could not read the API key for '{}' from the secret store: {e}",
            provider.id
        )),
    }
    if let Some(key) = non_empty(file_key.map(str::to_string)) {
        warn(
            "the plaintext 'api_key' in the config file is deprecated; \
             run `lumen configure` to move it to the secure store"
                .to_string(),
        );
        return Ok(Some(key));
    }

    Err(LumenError::MissingApiKey(format!(
        "no API key for {}. Checked: --api-key, LUMEN_API_KEY, {}, secret store \
         (service '{KEYCHAIN_SERVICE}', account '{}'), config file 'api_key'. \
         Run `lumen configure` to store a key.",
        provider.display_name, provider.env_key, provider.id
    )))
}

#[cfg(test)]
pub mod test_support {
    use super::SecretStore;
    use std::cell::RefCell;
    use std::collections::HashMap;

    /// In-memory secret store. `fail` makes every call return an error.
    #[derive(Default)]
    pub struct FakeStore {
        pub items: RefCell<HashMap<String, String>>,
        pub fail: Option<String>,
    }

    impl FakeStore {
        pub fn with(provider_id: &str, secret: &str) -> Self {
            let store = Self::default();
            store
                .items
                .borrow_mut()
                .insert(provider_id.to_string(), secret.to_string());
            store
        }

        pub fn failing(msg: &str) -> Self {
            Self {
                fail: Some(msg.to_string()),
                ..Self::default()
            }
        }
    }

    impl SecretStore for FakeStore {
        fn get(&self, id: &str) -> Result<Option<String>, String> {
            match &self.fail {
                Some(e) => Err(e.clone()),
                None => Ok(self.items.borrow().get(id).cloned()),
            }
        }
        fn set(&self, id: &str, secret: &str) -> Result<(), String> {
            match &self.fail {
                Some(e) => Err(e.clone()),
                None => {
                    self.items
                        .borrow_mut()
                        .insert(id.to_string(), secret.to_string());
                    Ok(())
                }
            }
        }
        fn delete(&self, id: &str) -> Result<bool, String> {
            match &self.fail {
                Some(e) => Err(e.clone()),
                None => Ok(self.items.borrow_mut().remove(id).is_some()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::FakeStore;
    use super::*;
    use crate::config::cli::ProviderType;
    use std::collections::HashMap;

    fn deepseek() -> &'static ProviderInfo {
        ProviderInfo::for_provider(ProviderType::Deepseek)
    }

    fn env_of(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let map: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |k| map.get(k).cloned()
    }

    fn resolve(
        flag: Option<&str>,
        file: Option<&str>,
        env: &[(&str, &str)],
        store: &FakeStore,
    ) -> (Result<Option<String>, LumenError>, Vec<String>) {
        let mut warnings = Vec::new();
        let env = env_of(env);
        let r = resolve_api_key(deepseek(), flag, file, &env, store, &mut |w| {
            warnings.push(w)
        });
        (r, warnings)
    }

    const ALL_ENV: &[(&str, &str)] = &[
        ("LUMEN_API_KEY", "lumen-env"),
        ("DEEPSEEK_API_KEY", "ds-env"),
    ];

    #[test]
    fn flag_wins() {
        let (r, w) = resolve(
            Some("flag"),
            Some("file"),
            ALL_ENV,
            &FakeStore::with("deepseek", "kc"),
        );
        assert_eq!(r.unwrap().as_deref(), Some("flag"));
        assert!(w.is_empty());
    }

    #[test]
    fn lumen_env_wins_over_provider_env() {
        let (r, _) = resolve(
            None,
            Some("file"),
            ALL_ENV,
            &FakeStore::with("deepseek", "kc"),
        );
        assert_eq!(r.unwrap().as_deref(), Some("lumen-env"));
    }

    #[test]
    fn provider_env_wins_over_keychain() {
        let env = &[("DEEPSEEK_API_KEY", "ds-env")];
        let (r, _) = resolve(None, Some("file"), env, &FakeStore::with("deepseek", "kc"));
        assert_eq!(r.unwrap().as_deref(), Some("ds-env"));
    }

    #[test]
    fn keychain_wins_over_config_file_without_warning() {
        let (r, w) = resolve(None, Some("file"), &[], &FakeStore::with("deepseek", "kc"));
        assert_eq!(r.unwrap().as_deref(), Some("kc"));
        assert!(w.is_empty());
    }

    #[test]
    fn keychain_item_of_other_provider_is_not_used() {
        let (r, _) = resolve(None, None, &[], &FakeStore::with("claude", "kc"));
        assert!(matches!(r, Err(LumenError::MissingApiKey(_))));
    }

    #[test]
    fn config_file_key_is_used_last_with_deprecation_warning() {
        let (r, w) = resolve(None, Some("file"), &[], &FakeStore::default());
        assert_eq!(r.unwrap().as_deref(), Some("file"));
        assert_eq!(w.len(), 1);
        assert!(w[0].contains("deprecated"));
    }

    #[test]
    fn keychain_failure_warns_and_continues() {
        let (r, w) = resolve(None, Some("file"), &[], &FakeStore::failing("user denied"));
        assert_eq!(r.unwrap().as_deref(), Some("file"));
        assert!(w[0].contains("user denied"));
    }

    #[test]
    fn empty_values_are_skipped() {
        let env = &[("LUMEN_API_KEY", ""), ("DEEPSEEK_API_KEY", " ")];
        let (r, _) = resolve(Some(""), None, env, &FakeStore::with("deepseek", "kc"));
        assert_eq!(r.unwrap().as_deref(), Some("kc"));
    }

    #[test]
    fn missing_key_error_names_every_source() {
        let (r, _) = resolve(None, None, &[], &FakeStore::default());
        let msg = r.unwrap_err().to_string();
        for part in [
            "--api-key",
            "LUMEN_API_KEY",
            "DEEPSEEK_API_KEY",
            "account 'deepseek'",
            "api_key",
        ] {
            assert!(msg.contains(part), "missing '{part}' in: {msg}");
        }
    }

    #[test]
    fn keyless_provider_needs_no_key_and_skips_store() {
        let ollama = ProviderInfo::for_provider(ProviderType::Ollama);
        let r = resolve_api_key(
            ollama,
            None,
            None,
            &|_| None,
            &FakeStore::failing("x"),
            &mut |w| panic!("unexpected warning: {w}"),
        );
        assert_eq!(r.unwrap(), None);
    }

    /// Touches the real OS store; run with `cargo test -- --ignored os_keychain`.
    #[test]
    #[ignore]
    fn os_keychain_round_trip() {
        let id = "lumen-selftest";
        let _ = OsKeychain.delete(id);
        assert_eq!(OsKeychain.get(id).unwrap(), None);
        OsKeychain.set(id, "secret-1").unwrap();
        assert_eq!(OsKeychain.get(id).unwrap().as_deref(), Some("secret-1"));
        assert!(OsKeychain.delete(id).unwrap());
        assert!(!OsKeychain.delete(id).unwrap());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_build_uses_a_real_platform_store() {
        assert!(keyring::Entry::store_status().is_ok());
    }
}
