use std::fmt::Display;

use crate::model::Model;
use crate::prelude::{InferIter, InferParams, ModelType};

/// Represents a chat between user and model.
pub struct Chat {
    model_type: ModelType,
    infer_iter: InferIter,
    /// The messages in the chat. This should be synchronized with the context of the infer_iter.
    chat_history: Vec<ChatMessage>,
}

impl Chat {
    /// Creates a new chat.
    pub fn new(
        model: &mut Model,
        system_prompt: impl AsRef<str>,
        chat_history: &[ChatMessage],
        infer_params: &InferParams,
    ) -> Self {
        // Create the initial context for the chat using the model's prompt template and tokenize it
        let full_prompt = model
            .model_type()
            .create_chat_prompt(system_prompt, chat_history, None);
        let initial_context = model.tokenize(full_prompt);

        // Create the InferIter for the chat with the initial context
        let infer_iter = model
            .infer_iter(
                initial_context,
                infer_params,
            )
            .unwrap();

        Self {
            model_type: model.model_type(),
            infer_iter,
            chat_history: chat_history.to_vec(),
        }
    }

    /// Push an existing message to the chat history.
    pub fn push_message(&mut self, message: ChatMessage) {
        // First add the complete message prompt to the message buffer
        self.infer_iter.push_str(
             self
                .model_type
                .create_chat_message_begin_prompt(message.sender()),
        );
        self.infer_iter.push_str(message.content());
        self.infer_iter.push_str("\n");
        self.infer_iter
            .push_str(self.model_type.create_chat_message_end_prompt());

        // Then push the message to the chat history
        self.chat_history.push(message);
    }

    /// Infer a new message from the model and push it to the chat history.
    /// If a begin_sequence is provided, it is treated as if it was prepended to the model's response, influencing the inference.
    pub fn infer_message(
        &mut self,
        sender: &ChatRole,
        begin_sequence: Option<&str>,
        end_sequences: &[&str],
    ) -> &str {
        // TODO: Compress and recreate the InferIter if chat tokens get too big to fit in the context window!!!
        // First add the beginning of the message prompt to the context
        self.infer_iter
            .push_str(self.model_type.create_chat_message_begin_prompt(sender));

        // If begin_sequence is Some, add it to the message buffer as well
        if let Some(begin_sequence) = begin_sequence {
            self.infer_iter.push_str(begin_sequence);
        }

        // Build end sequences
        let mut full_end_sequences = vec![self.model_type.chat_message_end_sequence()];
        full_end_sequences.extend_from_slice(end_sequences);

        // Then infer the response from the model, using the message buffer as the insert_before to inject new messages first
        let response = self
            .infer_iter
            .complete(&full_end_sequences);

        // Reset the message buffer to just the end message prompt
        self.infer_iter.push_str(self.model_type.create_chat_message_end_prompt());

        // Add the inferred response to the chat history as a new message
        let message = ChatMessage::new(sender.clone(), response.trim());
        self.chat_history.push(message);
        self.chat_history.last().unwrap().content()
    }

    /// Get the last message in the chat history, if any.
    pub fn last(&self) -> Option<&ChatMessage> {
        self.chat_history.last()
    }

    /// Consume the chat and return the last message's content, if any.
    pub fn into_last(mut self) -> Option<String> {
        // Move out just the last message then return its content
        self.chat_history.pop().map(|msg| msg.into_content())
    }
}

/// Represents a single message in a chat.
#[derive(Clone, Debug)]
pub struct ChatMessage {
    sender: ChatRole,
    content: String,
}

impl ChatMessage {
    /// Creates a new chat message.
    pub fn new(sender: ChatRole, content: impl Display) -> Self {
        Self { sender, content: content.to_string() }
    }

    /// Returns the sender of the message.
    pub fn sender(&self) -> &ChatRole {
        &self.sender
    }

    /// Returns the content of the message.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Consumes the message and returns its content.
    pub fn into_content(self) -> String {
        self.content
    }
}

impl Display for ChatMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.sender, self.content)
    }
}

impl Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::User => write!(f, "User"),
            ChatRole::Model => write!(f, "Model"),
            ChatRole::Other(name) => write!(f, "{}", name),
        }
    }
}

/// Represents the role of a participant in a chat.
#[derive(Clone, Debug)]
pub enum ChatRole {
    User,
    Model,
    Other(String),
}
