use std::collections::HashMap;

use crate::{chat_schema::SchemaWriteOutput, prelude::*};

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
    fn get_output(&mut self, input_context: &HashMap<String, String>) -> SchemaWriteOutput {
        let input_schema = self.input_schema().clone();
        let output_schema = self.output_schema().clone();
        let chat = self.chat_mut();

        // Add input message
        chat.add_message_with_infer_iter(&ChatRole::User, |infer_iter| {
            input_schema.write_input(infer_iter, input_context)
        });

        // Infer the output message using the output schema
        chat.add_message_with_infer_iter(&ChatRole::Assistant, |infer_iter| {
            output_schema.write_output(infer_iter)
        })
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
        infer_params: &InferParams,
        system_schema: impl Into<ChatSchema>,
        input_schema: impl Into<ChatSchema>,
        output_schema: impl Into<ChatSchema>,
        example_pairs: impl IntoIterator<Item = (HashMap<String, String>, HashMap<String, String>)>,
    ) -> Self {
        let system_schema = system_schema.into();
        let input_schema = input_schema.into();
        let output_schema = output_schema.into();
        let example_messages = create_chat_wrapper_examples(&input_schema, &output_schema, example_pairs);
        let chat = Chat::new(
            model,
            system_schema.to_input_string(&HashMap::new()),
            &example_messages,
            infer_params,
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
