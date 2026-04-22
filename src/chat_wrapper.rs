use std::collections::HashMap;

use crate::prelude::*;

/// Simulates a chat conversation by plugging given input contexts and corresponding outputs into the schemas.
pub fn create_example_chat_history(
    input_schema: &ChatSchema,
    output_schema: &ChatSchema,
    inputs: &[impl AsRef<HashMap<String, String>>],
    outputs: &[&[impl AsRef<str>]],
) -> Vec<ChatMessage> {
    let mut input_messages = Vec::new();
    let mut output_messages = Vec::new();

    // Populate the input messages
    for input_context in inputs {
        let input_context = input_context.as_ref();
        let input_message = ChatMessage::new(ChatRole::User, input_schema.to_input_string(input_context));
        input_messages.push(input_message);
    }

    // Populate the output messages
    for outputs in outputs {
        let output_message = ChatMessage::new(ChatRole::Assistant, output_schema.to_output_string(outputs));
        output_messages.push(output_message);
    }

    // Interleave the input and output messages to create the chat history
    input_messages
        .into_iter()
        .zip(output_messages)
        .flat_map(|(input_msg, output_msg)| vec![input_msg, output_msg])
        .collect()
}

/// Uses schema to define the structure of the input and output for chat-based interactions with a model.
pub trait ChatWrapper {
    /// The inference parameters to use for the chat wrapper.
    fn infer_params(&self) -> InferParams {
        InferParams::new_balanced()
    }
    /// Gets the input schema for the chat wrapper, which defines the structure of the user messages.
    fn input_schema(&self) -> &ChatSchema;
    /// Gets the output schema for the chat wrapper, which defines the structure of the assistant messages.
    fn output_schema(&self) -> &ChatSchema;
    /// Gets reference to the underlying chat.
    fn chat(&self) -> &Chat;
    /// Gets a mutable reference to the underlying chat.
    fn chat_mut(&mut self) -> &mut Chat;
    /// Gets the output of the chat wrapper for the given input context, using the internal schemas.
    /// Also returns the input string (as the second element of the tuple) that was generated from the input schema for reference.
    fn get_output(&mut self, input_context: &HashMap<String, String>) -> (String, String) {
        let input_schema = self.input_schema().clone();
        let output_schema = self.output_schema().clone();
        let chat = self.chat_mut();

        // Add input message
        let input = chat.add_message_with_infer_iter(&ChatRole::User, |infer_iter| {
            input_schema.write_input(infer_iter, input_context)
        });

        // Infer the output message using the output schema
        let output = chat.add_message_with_infer_iter(&ChatRole::Assistant, |infer_iter| {
            output_schema.write_output(infer_iter)
        });

        (output, input)
    }
}

/// A simple implementation of ChatWrapper that applies the given schemas directly without any additional logic.
pub struct SimpleChatWrapper {
    chat: Chat,
    input_schema: ChatSchema,
    output_schema: ChatSchema,
}

impl SimpleChatWrapper {
    /// Creates a new SimpleChatWrapper with the provided model and schemas.
    pub fn new(
        model: &mut Model,
        system_schema: impl Into<ChatSchema>,
        input_schema: impl Into<ChatSchema>,
        output_schema: impl Into<ChatSchema>,
    ) -> Self {
        let system_schema = system_schema.into();
        let input_schema = input_schema.into();
        let output_schema = output_schema.into();
        let chat = Chat::new(
            model,
            system_schema.to_input_string(&HashMap::new()),
            &[],
            &InferParams::new_logical(),
            None,
        );

        Self {
            chat,
            input_schema,
            output_schema,
        }
    }
}

impl ChatWrapper for SimpleChatWrapper {
    fn input_schema(&self) -> &ChatSchema {
        &self.input_schema
    }

    fn output_schema(&self) -> &ChatSchema {
        &self.output_schema
    }

    fn chat(&self) -> &Chat {
        &self.chat
    }

    fn chat_mut(&mut self) -> &mut Chat {
        &mut self.chat
    }
}
