use std::fmt::Display;

use aho_corasick::AhoCorasick;

use crate::{
    chat::{Chat, ChatMessage, ChatRole},
    data::{JsonMap, JsonValue},
    model::Model,
};

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

/// A single step in a pipeline.
#[derive(Clone, Debug)]
enum PipelineStep {
    Chat {
        output_key: String,
        system_prompt: String,
        chat_history: Vec<ChatMessage>,
        user_prompt: String,
        end_sequences: Vec<String>,
        response_prefix: String,
        seed: u64,
        temp: f64,
    },
}

impl PipelineStep {
    /// Called when the pipeline is executed. Takes in the current context and returns any new key-value pairs to add to the context.
    fn execute(&self, model: &Model, context: &mut JsonMap) {
        match self {
            PipelineStep::Chat {
                output_key,
                system_prompt,
                chat_history,
                user_prompt,
                end_sequences,
                response_prefix,
                seed,
                temp,
            } => {
                // Substitute context keys in the passed strings
                let system_prompt = substitute_context_keys(system_prompt, context);
                let user_prompt = substitute_context_keys(user_prompt, context);
                let chat_history = chat_history
                    .iter()
                    .map(|message| {
                        let content = substitute_context_keys(message.content(), context);
                        ChatMessage::new(message.sender().clone(), content)
                    })
                    .collect::<Vec<ChatMessage>>();
                let response_prefix = substitute_context_keys(response_prefix, context);

                // Convert end_sequences to a Vec<&str> for the complete method
                let mut end_sequences = end_sequences
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<&str>>();

                // Append a "---" end sequence if none of the provided end sequences contain it
                // This helps avoid unnecessary tokens being generated after the model's response in some cases
                if !end_sequences.iter().any(|s| s.contains("---")) {
                    end_sequences.push("---");
                }

                // Build the chat
                let mut chat = Chat::new(model, system_prompt, &chat_history, *seed, Some(*temp));
                chat.push_message(ChatMessage::new(ChatRole::User, user_prompt));

                // Infer the model's response with the provided end sequences and response prefix
                let response =
                    chat.infer_message(&ChatRole::Model, &end_sequences, Some(response_prefix));

                // Store the response in the context under output_key
                context.insert(output_key.clone(), JsonValue::String(response.to_string()));
            }
        }
    }
}

/// Represents a full pipeline which takes in inputs and produces transformed outputs.
/// Transformation steps can include generating & storing text, summarizing steps, etc.
#[derive(Clone, Debug)]
pub struct Pipeline {
    seed: u64,
    steps: Vec<PipelineStep>,
}

impl Pipeline {
    /// Create a new empty pipeline.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            steps: Vec::new(),
        }
    }

    /// Get the next seed and increment the pipeline's seed for the next step.
    pub fn next_seed(&mut self) -> u64 {
        let current_seed = self.seed;
        self.seed = self.seed.wrapping_add(1);
        current_seed
    }

    /// Add a chat step to the pipeline with the given parameters. The chat's response will be stored in the context under `output_key`.
    pub fn chat(
        &mut self,
        output_key: impl Display,
        system_prompt: impl Display,
        chat_history: impl Into<Vec<ChatMessage>>,
        user_prompt: impl Display,
        response_prefix: impl Display,
        end_sequences: impl Into<Vec<String>>,
        temp: f64,
    ) -> &mut Self {
        let seed = self.next_seed();
        let chat_history = chat_history.into();

        self.steps.push(PipelineStep::Chat {
            output_key: output_key.to_string(),
            system_prompt: system_prompt.to_string(),
            chat_history,
            user_prompt: user_prompt.to_string(),
            response_prefix: response_prefix.to_string(),
            end_sequences: end_sequences
                .into()
                .into_iter()
                .map(|s| s.to_string())
                .collect(),
            seed,
            temp,
        });

        self
    }

    /// Add a summarization step to the pipeline which summarizes the text stored under `input_key` and stores the summary under `output_key`.
    /// The summarization is done by prompting the model to summarize the text, with an optional hint to guide the summarization.
    pub fn summarize(
        &mut self,
        output_key: impl Display,
        input_key: impl Display,
        hint: Option<String>,
    ) -> &mut Self {
        // Build the system prompt for the summarization step, incorporating the hint if provided
        let system_prompt = if let Some(hint) = hint {
            format!("You are a helpful assistant who summarizes text. {}", hint,)
        } else {
            "You are a helpful assistant who summarizes long text to make it shorter.".to_string()
        };

        // Ask the model to summarize the text under input_key
        let user_prompt = format!(
            "Please summarize the text to make it shorter!\n\n**Text to summarize:**\n{{{}}}",
            input_key
        );

        self.chat(
            output_key,
            system_prompt,
            [],
            user_prompt,
            "**Summary:**\n",
            [],
            0.5,
        );

        self
    }

    /// Run the pipeline until completion and return the final context as a map of string keys to JSON values.
    pub fn execute(&self, model: &Model, input: impl Into<JsonMap>) -> JsonMap {
        // Start the context with the provided inputs
        let mut context = input.into();

        // Execute each step in the pipeline, allowing them to read from and write to the context
        for step in &self.steps {
            step.execute(model, &mut context);
        }

        context
    }
}

impl Default for Pipeline {
    fn default() -> Self {
        Self::new(123456789)
    }
}
