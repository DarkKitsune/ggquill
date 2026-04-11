/*use std::fmt::Display;

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

/// A pipeline is a sequence of steps that can be executed to generate a response.
pub struct Pipeline {
    steps: Vec<PipelineStep>,
}

impl Pipeline {
    pub fn new(steps: Vec<PipelineStep>) -> Self {
        Self { steps }
    }

    /// Execute the pipeline with the given model and context, returning the final response.
    pub fn execute(&self, model: &Model, context: &JsonMap) -> String {
        let mut chat = Chat::new();

        for step in &self.steps {

        
    }
}

/// A step in the pipeline can be either a chat message or a function call.
pub enum PipelineStep {
    instruction(String),
}*/