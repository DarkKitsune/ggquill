use std::fmt::Display;

use aho_corasick::AhoCorasick;
use anyhow::Result;

use crate::{
    chat::{Chat, ChatMessage, ChatRole},
    data::{JsonMap, JsonValue}, model::Model, prelude::InferParams,
};

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant. You answer questions and follow instructions directly, \
    concisely, and accurately, in the user's language. Avoid excessive pleasantries, repetitions, or repeating the question.";

/// Helper function for parsing pipeline context keys in the format "{key}" and substituting
/// them with their JSON values from the context.
fn substitute_context_keys(input: impl AsRef<str>, context: &JsonMap) -> String {
    let input = input.as_ref();

    // Build the patterns and replacements for the Aho-Corasick algorithm
    let (patterns, replacements): (Vec<String>, Vec<String>) = context
        .iter()
        .map(|(k, v)| (format!("{{{}}}", k), v.to_string()))
        .unzip();

    // Instead of a simple replace we use aho-corasick to find and replace all context keys in one pass
    let ac = AhoCorasick::new(patterns).unwrap();
    ac.replace_all(input, &replacements)
}

/// A step in the pipeline.
pub enum PipelineStep {
    SystemPrompt(String),
    Instruct {
        result_key: String,
        instruction: String,
        begin_sequence: Option<String>,
        end_sequences: Vec<String>,
    },
    Summarize {
        result_key: String,
        to_summarize: String,
        style_hint: String,
    },
}

impl PipelineStep {
    /// Execute the pipeline step with the given model and context, returning the response.
    pub fn execute(&self, chat: &mut Chat, context: &mut JsonMap) -> Result<()> {
        // Match self to determine the type of pipeline step and execute accordingly
        // If a result needs to be stored in the context then it will be stored in `result`
        let result = match self {
            PipelineStep::SystemPrompt(prompt) => {
                // Substitute context keys in the prompt
                let prompt = substitute_context_keys(prompt, context);

                // Add the system prompt as a system message to the chat
                chat.push_message(ChatMessage::new(ChatRole::Other("system".to_string()), prompt.clone()));
                None
            }
            PipelineStep::Instruct {
                instruction,
                begin_sequence,
                end_sequences,
                ..
            } => {
                // Substitute context keys
                let instruction = substitute_context_keys(instruction, context);
                let begin_sequence = begin_sequence.as_ref().map(|s| substitute_context_keys(s, context));
                let end_sequences = end_sequences.iter().map(|s| substitute_context_keys(s, context)).collect::<Vec<_>>();

                // Add the instruction as a user message to the chat
                chat.push_message(ChatMessage::new(ChatRole::User, instruction.clone()));

                // Infer a response from the model using the instruction as a prompt
                Some(chat.infer_message(
                    &ChatRole::Model,
                    begin_sequence.as_deref(),
                    &end_sequences.iter().map(String::as_str).collect::<Vec<_>>(),
                ).trim().to_string())
            }

            PipelineStep::Summarize { style_hint, to_summarize, .. } => {
                // Substitute context keys
                let style_hint = substitute_context_keys(style_hint, context);
                let to_summarize = substitute_context_keys(to_summarize, context);

                // Write the instruction
                let style_prompt = if !style_hint.is_empty() {
                    format!("\n\n**The summary should have the following characteristic(s):**\n{}", style_hint)
                } else {
                    "".to_string()
                };
                let instruction = format!(
                    "**Summarize the following text:**\n\"{}\"{}",
                    to_summarize,
                    style_prompt,
                );

                // Add the instruction as a user message to the chat
                chat.push_message(ChatMessage::new(ChatRole::User, instruction));

                // Infer a response from the model using the instruction as a prompt
                Some(chat.infer_message(
                    &ChatRole::Model,
                    Some("**Here is the summary:**\n"),
                    &[],
                ).trim().to_string())
            }
        };

        // Store the response in the context under the result key
        if let Some(result) = result && let Some(result_key) = self.result_key() {
            context.insert(result_key.clone(), JsonValue::String(result.to_string()));
        }

        Ok(())
    }

    /// Get the result key for the pipeline step, if any.
    /// If the step produces a result that should be stored in the context, this is the key under which it should be stored.
    fn result_key(&self) -> Option<&String> {
        match self {
            PipelineStep::SystemPrompt(_) => None,
            PipelineStep::Instruct { result_key, .. } => Some(result_key),
            PipelineStep::Summarize { result_key, .. } => Some(result_key),
        }
    }
}

/// A pipeline is a sequence of steps that can be executed to generate a response.
/// Internally a chat is used, and the pipeline guides the prompts and validates responses, and replaces context keys.
pub struct Pipeline {
    chat: Chat,
    steps: Vec<PipelineStep>,
    persistent_memory: bool,
    has_executed: bool,
}

impl Pipeline {
    /// Creates a new pipeline with the provided model. The pipeline starts out empty.
    pub fn new(model: &mut Model) -> Self {
        let chat = Chat::new(
            model,
            DEFAULT_SYSTEM_PROMPT,
            &[],
            &InferParams::new_balanced(),
        );
        Self { chat, steps: Vec::new(), persistent_memory: true, has_executed: false }
    }

    /// Sets whether the pipeline should use persistent memory.
    /// If true, then the cleanup step is skipped, and subsequent executions of the pipeline may be influenced by previous ones.
    pub fn set_persistent_memory(&mut self, persistent: bool) {
        self.persistent_memory = persistent;
    }

    /// Gets whether the pipeline is using persistent memory.
    pub fn is_persistent_memory(&self) -> bool {
        self.persistent_memory
    }

    /// Add a pipeline step to the end of the pipeline.
    pub fn add_step(&mut self, step: PipelineStep) {
        self.steps.push(step);
    }

    /// Get a reference to the chat used internally by the pipeline.
    pub fn chat(&self) -> &Chat {
        &self.chat
    }

    /// Execute the pipeline with the given context, returning the final response.
    pub fn execute(&mut self, context: &mut JsonMap) {
        // If we have already executed, and persistent_memory is false, then reset the chat
        if self.has_executed && !self.persistent_memory {
            self.chat.reset(DEFAULT_SYSTEM_PROMPT, vec![]);
        }
        self.has_executed = true;

        // Execute each step in the pipeline sequentially, allowing them to modify the chat and context as needed
        for step in &self.steps {
            step.execute(&mut self.chat, context).unwrap();
        }
    }

    /// Add a system prompt step to the pipeline. This is used to direct the model's behavior in subsequent steps.
    /// Returns a mutable reference to the pipeline to allow for chaining.
    pub fn system_prompt(&mut self, prompt: impl Display) -> &mut Self {
        self.add_step(PipelineStep::SystemPrompt(prompt.to_string()));
        self
    }

    /// Add an instruct step.
    /// Returns a mutable reference to the pipeline to allow for chaining.
    pub fn instruct(
        &mut self,
        result_key: impl Display,
        instruction: impl Display,
        begin_sequence: Option<String>,
        end_sequences: Vec<String>,
    ) -> &mut Self {
        self.add_step(PipelineStep::Instruct {
            result_key: result_key.to_string(),
            instruction: instruction.to_string(),
            begin_sequence,
            end_sequences,
        });

        self
    }

    /// Add a summarize step.
    /// Returns a mutable reference to the pipeline to allow for chaining.
    /// The style_hint is a string that describes the desired characteristics, such as "concise and easy to understand"
    pub fn summarize(
        &mut self,
        result_key: impl Display,
        to_summarize: impl Display,
        style_hint: impl Display,
    ) -> &mut Self {
        self.add_step(PipelineStep::Summarize {
            result_key: result_key.to_string(),
            to_summarize: to_summarize.to_string(),
            style_hint: style_hint.to_string(),
        });
        self
    }
}