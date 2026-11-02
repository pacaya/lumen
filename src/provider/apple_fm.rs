//! Apple Foundation Models through the `fm respond` CLI (no server needed).
use std::io::ErrorKind;
use std::process::Stdio;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use super::ProviderError;

const CONTEXT_EXCEEDED: &str = "exceeded the model's context size";

/// Runs `fm respond --no-stream -m <model> -i <instructions>` with `prompt` on stdin.
pub async fn complete(
    program: &str,
    model: &str,
    instructions: &str,
    prompt: &str,
) -> Result<String, ProviderError> {
    let mut child = Command::new(program)
        .args(["respond", "--no-stream", "-m", model, "-i", instructions])
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| match e.kind() {
            ErrorKind::NotFound => ProviderError::AppleFmNotFound(program.to_string()),
            _ => ProviderError::AppleFmFailed(e.to_string()),
        })?;

    if let Some(mut stdin) = child.stdin.take() {
        // `fm` can exit before it reads all input (for example on an error).
        // The exit status below reports that case.
        match stdin.write_all(prompt.as_bytes()).await {
            Ok(()) => {}
            Err(e) if e.kind() == ErrorKind::BrokenPipe => {}
            Err(e) => return Err(ProviderError::AppleFmFailed(e.to_string())),
        }
    }

    let output = child
        .wait_with_output()
        .await
        .map_err(|e| ProviderError::AppleFmFailed(e.to_string()))?;

    if !output.status.success() {
        let stderr = strip_ansi(&String::from_utf8_lossy(&output.stderr));
        if stderr.contains(CONTEXT_EXCEEDED) {
            return Err(ProviderError::AppleContextExceeded(context_hint(model)));
        }
        let msg = stderr.trim().trim_start_matches("Error:").trim();
        return Err(ProviderError::AppleFmFailed(if msg.is_empty() {
            format!("fm exited with {}", output.status)
        } else {
            msg.to_string()
        }));
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if text.is_empty() {
        return Err(ProviderError::NoCompletionChoice);
    }
    Ok(text)
}

/// Removes ANSI CSI escape sequences. `fm` colors stderr even when piped.
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if ('@'..='~').contains(&c) {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn context_hint(model: &str) -> String {
    if model == "system" {
        "the input is too large for the on-device Apple model. \
         Use `-m pcc` (Private Cloud Compute, needs iCloud sign-in) or another provider."
            .to_string()
    } else {
        format!("the input is too large for Apple model '{model}'. Use another provider.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn fake_fm(body: &str) -> (tempfile::TempDir, String) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fm");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        let p = path.to_str().unwrap().to_string();
        (dir, p)
    }

    #[tokio::test]
    async fn passes_model_and_instructions_as_args_and_prompt_on_stdin() {
        let (_d, fm) = fake_fm(r#"printf '%s|' "$@"; printf 'stdin='; cat"#);
        let out = complete(&fm, "pcc", "be brief", "the diff").await.unwrap();
        assert_eq!(out, "respond|--no-stream|-m|pcc|-i|be brief|stdin=the diff");
    }

    #[tokio::test]
    async fn context_overflow_suggests_pcc_for_system_model() {
        let (_d, fm) = fake_fm(
            r#"cat >/dev/null; echo "Error: The session's transcript exceeded the model's context size." >&2; exit 1"#,
        );
        let err = complete(&fm, "system", "i", "big").await.unwrap_err();
        assert!(matches!(err, ProviderError::AppleContextExceeded(ref m) if m.contains("-m pcc")));
    }

    #[tokio::test]
    async fn context_overflow_on_pcc_suggests_another_provider() {
        let (_d, fm) = fake_fm(r#"echo "Error: exceeded the model's context size." >&2; exit 1"#);
        let err = complete(&fm, "pcc", "i", "big").await.unwrap_err();
        let ProviderError::AppleContextExceeded(m) = err else {
            panic!()
        };
        assert!(!m.contains("-m pcc") && m.contains("another provider"));
    }

    #[tokio::test]
    async fn other_failures_report_fm_stderr() {
        let (_d, fm) = fake_fm(
            r#"echo "Error: Using Private Cloud Compute requires an iCloud account." >&2; exit 1"#,
        );
        let err = complete(&fm, "pcc", "i", "p").await.unwrap_err();
        assert!(
            matches!(err, ProviderError::AppleFmFailed(ref m) if m.starts_with("Using Private Cloud"))
        );
    }

    #[tokio::test]
    async fn colored_stderr_is_stripped() {
        let (_d, fm) =
            fake_fm(r#"printf 'Error: \033[38;2;255;107;128mNeeds iCloud.\033[0m\n' >&2; exit 1"#);
        let err = complete(&fm, "pcc", "i", "p").await.unwrap_err();
        assert!(
            matches!(err, ProviderError::AppleFmFailed(ref m) if m == "Needs iCloud."),
            "{err}"
        );
    }

    #[tokio::test]
    async fn missing_fm_is_a_clear_error() {
        let err = complete("/nonexistent/fm", "system", "i", "p")
            .await
            .unwrap_err();
        assert!(matches!(err, ProviderError::AppleFmNotFound(_)));
    }

    #[tokio::test]
    async fn empty_output_is_no_completion() {
        let (_d, fm) = fake_fm("cat >/dev/null");
        let err = complete(&fm, "system", "i", "p").await.unwrap_err();
        assert!(matches!(err, ProviderError::NoCompletionChoice));
    }
}
