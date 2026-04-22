use std::fmt::Display;

use crate::prelude::*;

/// Used to join together many values that have string representations into a single string.
/// The provided model is used to fill in the gap and correct grammar/formatting.
pub struct Joiner {
    chat: Chat,
}

impl Joiner {
    /// Creates a new Joiner with the provided model, seed, and temperature.
    pub fn new(model: &mut Model) -> Self {
        let example_chat_history = [
            ChatMessage::new(ChatRole::User, "an apple | orange | banana"),
            ChatMessage::new(ChatRole::Assistant, "\"an apple, an orange, and a banana\""),
            ChatMessage::new(ChatRole::User, "Please inform | Jack | meeting | 3PM | !"),
            ChatMessage::new(
                ChatRole::Assistant,
                "\"Please inform Jack about the meeting at 3:00 PM!\"",
            ),
            ChatMessage::new(ChatRole::User, "I am | drive | red | big | car."),
            ChatMessage::new(ChatRole::Assistant, "\"I am driving a big red car.\""),
        ];

        let chat = Chat::new(
            model,
            "You are a helpful writing assistant. Your goal is to take a list of input strings or values \
            separated by '|' and join them together into a single string. The final output should be a well \
            formatted and grammatically correct string in double quotes which incorporates all of the input \
            values. Do not add extra symbols or punctuation other than that which is in the input.",
            &example_chat_history,
            &InferParams::new_logical(),
            None,
        );
        Self { chat }
    }

    /// Joins the provided values into a single string using the model. The input values will be separated by '|'.
    pub fn join(&mut self, values: &[impl Display]) -> &str {
        // Format the input values into a single string separated by '|'
        let input = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" | ");

        // Push the input as a user message to the chat
        self.chat
            .push_message(ChatMessage::new(ChatRole::User, input));

        // Infer a model response with the input and return the result
        self.chat
            .infer_message(&ChatRole::Assistant, Some("\""), &["\""])
    }
}