//! Keeps AI prompts inside an input-token budget.
//!
//! The token count is an estimate: characters ÷ [`CHARS_PER_TOKEN`]. When a
//! prompt is over budget, the largest file bodies in the diff are replaced by a
//! one-line summary until the prompt fits.

/// Default input-token budget when `providers.<id>.max_input_tokens` is not set.
pub const DEFAULT_MAX_INPUT_TOKENS: usize = 100_000;

/// Characters per token used for the estimate.
pub const CHARS_PER_TOKEN: usize = 4;

/// Characters reserved for the omitted-files note, in addition to the file paths.
const NOTE_RESERVE: usize = 300;

/// Estimated tokens for a text of `chars` characters.
pub fn estimate_tokens(chars: usize) -> usize {
    chars.div_ceil(CHARS_PER_TOKEN)
}

/// A diff reduced to fit a character budget.
#[derive(Debug, PartialEq)]
pub struct FittedDiff {
    pub text: String,
    /// Paths whose diff content was replaced by a summary line.
    pub omitted: Vec<String>,
}

/// One `diff --git` section of a unified diff.
struct FileSection<'a> {
    text: &'a str,
    chars: usize,
}

impl FileSection<'_> {
    /// Path from the `+++ b/` line, then `--- a/`, then the `diff --git` header.
    fn path(&self) -> String {
        let lines = || self.text.lines();
        lines()
            .find_map(|l| l.strip_prefix("+++ b/"))
            .or_else(|| lines().find_map(|l| l.strip_prefix("--- a/")))
            .map(str::to_string)
            .or_else(|| {
                let header = lines().next()?.strip_prefix("diff --git a/")?;
                let (_, b) = header.split_once(" b/")?;
                Some(b.to_string())
            })
            .unwrap_or_default()
    }

    /// `<path>: +X -Y lines (diff omitted)`, with the status of added or deleted files.
    fn summary(&self) -> String {
        let mut added = 0;
        let mut removed = 0;
        let mut in_hunk = false;
        let mut status = "";
        for line in self.text.lines() {
            if line.starts_with("@@") {
                in_hunk = true;
            } else if !in_hunk {
                if line.starts_with("new file mode") {
                    status = " (new file)";
                } else if line.starts_with("deleted file mode") {
                    status = " (deleted)";
                }
            } else if line.starts_with('+') {
                added += 1;
            } else if line.starts_with('-') {
                removed += 1;
            }
        }
        format!(
            "{}{status}: +{added} -{removed} lines (diff omitted)\n",
            self.path()
        )
    }
}

/// Split a unified diff into the text before the first file and the file sections.
fn split_sections(diff: &str) -> (&str, Vec<FileSection<'_>>) {
    let mut starts: Vec<usize> = Vec::new();
    let mut offset = 0;
    for line in diff.split_inclusive('\n') {
        if line.starts_with("diff --git ") {
            starts.push(offset);
        }
        offset += line.len();
    }
    let Some(&first) = starts.first() else {
        return (diff, Vec::new());
    };
    starts.push(diff.len());
    let sections = starts
        .windows(2)
        .map(|w| {
            let text = &diff[w[0]..w[1]];
            FileSection {
                text,
                chars: text.chars().count(),
            }
        })
        .collect();
    (&diff[..first], sections)
}

/// Fit `diff` into `max_chars` characters.
///
/// File bodies are replaced by summary lines, largest first, until the diff
/// plus the omitted-files note fits. Returns `None` when even the summary of
/// every file does not fit.
pub fn fit_diff(diff: &str, max_chars: usize) -> Option<FittedDiff> {
    let (preamble, sections) = split_sections(diff);
    let mut parts: Vec<String> = sections.iter().map(|s| s.text.to_string()).collect();
    let mut total = preamble.chars().count() + sections.iter().map(|s| s.chars).sum::<usize>();

    if total <= max_chars {
        return Some(FittedDiff {
            text: diff.to_string(),
            omitted: Vec::new(),
        });
    }

    let mut order: Vec<usize> = (0..sections.len()).collect();
    order.sort_by(|&a, &b| sections[b].chars.cmp(&sections[a].chars));

    let mut omitted_idx: Vec<usize> = Vec::new();
    let mut note = NOTE_RESERVE;
    for idx in order {
        let summary = sections[idx].summary();
        let summary_chars = summary.chars().count();
        if summary_chars >= sections[idx].chars {
            continue; // Already as short as a summary (for example a binary entry).
        }
        total = total - sections[idx].chars + summary_chars;
        note += sections[idx].path().chars().count() + 2;
        parts[idx] = summary;
        omitted_idx.push(idx);
        if total + note <= max_chars {
            break;
        }
    }

    if total + note > max_chars {
        return None;
    }

    omitted_idx.sort_unstable();
    Some(FittedDiff {
        text: format!("{preamble}{}", parts.concat()),
        omitted: omitted_idx.iter().map(|&i| sections[i].path()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(path: &str, body_lines: usize) -> String {
        let mut s = format!(
            "diff --git a/{path} b/{path}\nindex 111..222 100644\n--- a/{path}\n+++ b/{path}\n@@ -1,1 +1,{body_lines} @@\n-old\n"
        );
        for i in 0..body_lines {
            s.push_str(&format!("+line {i} of {path}\n"));
        }
        s
    }

    #[test]
    fn estimate_rounds_up() {
        assert_eq!(estimate_tokens(0), 0);
        assert_eq!(estimate_tokens(1), 1);
        assert_eq!(estimate_tokens(8), 2);
        assert_eq!(estimate_tokens(9), 3);
    }

    #[test]
    fn diff_within_budget_is_unchanged() {
        let diff = file("a.rs", 3);
        let fitted = fit_diff(&diff, diff.len()).unwrap();
        assert_eq!(fitted.text, diff);
        assert!(fitted.omitted.is_empty());
    }

    #[test]
    fn largest_file_is_omitted_first_and_small_files_keep_content() {
        let small = file("small.rs", 2);
        let big = file("big.js", 500);
        let diff = format!("{small}{big}");
        let fitted = fit_diff(&diff, small.len() + 1_000).unwrap();

        assert_eq!(fitted.omitted, vec!["big.js".to_string()]);
        assert!(fitted.text.contains(&small), "small file keeps content");
        assert!(fitted
            .text
            .contains("big.js: +500 -1 lines (diff omitted)\n"));
        assert!(!fitted.text.contains("line 7 of big.js"));
    }

    #[test]
    fn summary_names_added_and_deleted_files() {
        let added = "diff --git a/n.txt b/n.txt\nnew file mode 100644\n--- /dev/null\n+++ b/n.txt\n@@ -0,0 +1,2 @@\n+a\n+b\n";
        let deleted = "diff --git a/d.txt b/d.txt\ndeleted file mode 100644\n--- a/d.txt\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-x\n";
        let (_, sections) = split_sections(added);
        assert_eq!(
            sections[0].summary(),
            "n.txt (new file): +2 -0 lines (diff omitted)\n"
        );
        let (_, sections) = split_sections(deleted);
        assert_eq!(
            sections[0].summary(),
            "d.txt (deleted): +0 -1 lines (diff omitted)\n"
        );
    }

    #[test]
    fn file_list_over_budget_returns_none() {
        let diff = format!("{}{}", file("a.rs", 100), file("b.rs", 100));
        assert!(fit_diff(&diff, 50).is_none());
    }

    #[test]
    fn binary_entries_are_kept_as_they_are() {
        let binary = "diff --git a/logo.png b/logo.png\nnew file mode 100644\nindex 0000000..1111111\nBinary files /dev/null and b/logo.png differ\n";
        let diff = format!("{binary}{}", file("big.rs", 500));
        let fitted = fit_diff(&diff, binary.len() + 600).unwrap();
        assert!(fitted.text.starts_with(binary));
        assert_eq!(fitted.omitted, vec!["big.rs".to_string()]);
    }
}
