use crate::{
    command::{draft::DraftCommand, explain::ExplainCommand},
    git_entity::{diff::Diff, GitEntity},
};
use indoc::{formatdoc, indoc};
use thiserror::Error;

#[derive(Error, Debug)]
#[error("{0}")]
pub struct AIPromptError(String);

#[derive(Debug)]
pub struct AIPrompt {
    pub system_prompt: String,
    pub user_prompt: String,
}

/// The diff text put into a prompt, and the files whose content it omits.
pub struct PromptDiff<'a> {
    pub text: &'a str,
    pub omitted: &'a [String],
}

impl<'a> PromptDiff<'a> {
    /// The complete diff, with nothing omitted.
    pub fn full(text: &'a str) -> Self {
        PromptDiff { text, omitted: &[] }
    }

    /// Tells the model which files have no content in the diff.
    fn omitted_note(&self) -> String {
        if self.omitted.is_empty() {
            return String::new();
        }
        formatdoc! {"

            Note: the prompt was too large, so the diff content of these files is omitted and only their names and line counts are shown: {files}.
            Do not guess the content of these files. If a question needs it, say that the content is not available.
            ",
            files = self.omitted.join(", ")
        }
    }
}

impl AIPrompt {
    /// Characters in the system and user prompts, for the token estimate.
    pub fn chars(&self) -> usize {
        self.system_prompt.chars().count() + self.user_prompt.chars().count()
    }

    pub fn build_explain_prompt(
        command: &ExplainCommand,
        diff: &PromptDiff,
    ) -> Result<Self, AIPromptError> {
        let system_prompt = String::from(indoc! {"
            You are a helpful assistant that explains Git changes in a concise way.
            Focus only on the most significant changes and their direct impact.
            When answering specific questions, address them directly and precisely.
            Keep explanations brief but informative and don't ask for further explanations.
            Use markdown for clarity.
        "});

        let base_content = match &command.git_entity {
            GitEntity::Commit(commit) => {
                formatdoc! {"
                    Context - Commit:

                    Message: {msg}
                    Changes:
                    ```diff
                    {diff}
                    ```
                    {note}",
                    msg = commit.message,
                    diff = diff.text,
                    note = diff.omitted_note(),
                }
            }
            GitEntity::Diff(_) => {
                formatdoc! {"
                    Context - Changes:

                    ```diff
                    {diff}
                    ```
                    {note}",
                    diff = diff.text,
                    note = diff.omitted_note(),
                }
            }
        };

        let user_prompt = match &command.query {
            Some(query) => {
                formatdoc! {"
                    {base_content}

                    Question: {query}

                    Provide a focused answer to the question based on the changes shown above.
                    "
                }
            }
            None => match &command.git_entity {
                GitEntity::Commit(_) => formatdoc! {"
                    {base_content}
                    
                    Provide a short explanation covering:
                    1. Core changes made
                    2. Direct impact
                    "
                },
                GitEntity::Diff(Diff::WorkingTree { .. }) => formatdoc! {"
                    {base_content}
                    
                    Provide:
                    1. Key changes
                    2. Notable concerns (if any)
                    "
                },
                GitEntity::Diff(Diff::CommitsRange { .. }) => formatdoc! {"
                    {base_content}
                    
                    Provide:
                    1. Core changes made
                    2. Direct impact
                    "
                },
            },
        };

        Ok(AIPrompt {
            system_prompt,
            user_prompt,
        })
    }

    pub fn build_draft_prompt(
        command: &DraftCommand,
        diff: &PromptDiff,
    ) -> Result<Self, AIPromptError> {
        let GitEntity::Diff(Diff::WorkingTree { .. }) = &command.git_entity else {
            return Err(AIPromptError(
                "`draft` is only supported for working tree diffs".into(),
            ));
        };

        let system_prompt = String::from(indoc! {"
            You are a commit message generator that follows these rules:
            1. Write in present tense
            2. Be concise and direct
            3. Output only the commit message without any explanations
            4. Follow the format: <type>(<optional scope>): <commit message>
        "});

        let context = if let Some(context) = &command.context {
            formatdoc!(
                "
                Use the following context to understand intent:
                {context}
                "
            )
        } else {
            "".to_string()
        };

        let user_prompt = formatdoc! {"
            Generate a concise git commit message written in present tense for the following code diff with the given specifications below:

            The output response must be in format:
            <type>(<optional scope>): <commit message>
            Choose a type from the type-to-description JSON below that best describes the git diff:
            {commit_types}
            Focus on being accurate and concise.
            {context}
            Commit message must be a maximum of 72 characters.
            Exclude anything unnecessary such as translation. Your entire response will be passed directly into git commit.

            Code diff:
            ```diff
            {diff}
            ```
            {note}",
            commit_types = command.draft_config.commit_types,
            diff = diff.text,
            note = diff.omitted_note(),
        };

        Ok(AIPrompt {
            system_prompt,
            user_prompt,
        })
    }

    pub fn build_operate_prompt(query: &str) -> Result<Self, AIPromptError> {
        let system_prompt = String::from(indoc! {"
        You're a Git assistant that provides commands with clear explanations.
        - Include warnings ONLY for destructive commands (reset, push --force, clean, etc.)
        - Omit warning tag completely for safe commands
    "});
        let user_prompt = formatdoc! {"
        Generate Git command for: {query}
        
        <command>Git command</command>
        <explanation>Brief explanation</explanation>
        <warning>Required for destructive commands only - omit for safe commands</warning>
        ",
            query = query
        };
        Ok(AIPrompt {
            system_prompt,
            user_prompt,
        })
    }
}
