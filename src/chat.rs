use std::fmt::Display;

use crate::data::{JsonMap, JsonValue};
use crate::model::{MAX_TOKENS, Model};
use crate::prelude::{InferIter, InferParams, ModelType};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant that tries to help the user with their requests as best as you can. \
    You should always try to be as helpful and informative as possible, while also being concise and clear in your responses. \
    Avoid unnecessary repetition and fluff, and try to get to the point quickly.";
const COMPRESS_MEMORY_THRESHOLD: usize = MAX_TOKENS / 2;
const ESTIMATE_CHARACTERS_PER_TOKEN: usize = 4;

/// Represents a chat between user and model.
pub struct Chat {
    model_type: ModelType,
    infer_iter: InferIter,
    /// The messages in the chat. This should be synchronized with the context of the infer_iter.
    chat_history: Vec<ChatMessage>,
    /// When there are too many messages we summarize the long_term_memory and half of the chat history together,
    /// store the summary in long_term_memory.
    long_term_memory: Option<String>,
    extra_data: Option<JsonMap>,
}

impl Chat {
    /// Creates a new chat.
    pub fn new(
        model: &mut Model,
        system_prompt: impl AsRef<str>,
        chat_history: &[ChatMessage],
        infer_params: &InferParams,
        extra_data: Option<JsonMap>,
    ) -> Self {
        // Create the initial context for the chat using the model's prompt template and tokenize it
        let full_prompt = model
            .model_type()
            .create_chat_prompt(system_prompt, chat_history, extra_data.as_ref());
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
            long_term_memory: None,
            extra_data,
        }
    }

    /// Resets the chat using a given system prompt and chat history.
    pub fn reset(&mut self, system_prompt: impl AsRef<str>, chat_history: Vec<ChatMessage>, long_term_memory: Option<String>) {
        // If long term memory is provided, set it in key "memory" of self.extra_data for the prompt template
        // Also create self.extra_data if it doesn't exist yet
        if let Some(long_term_memory) = &long_term_memory {
            let extra_data = self.extra_data.get_or_insert_with(JsonMap::new);
            extra_data.insert("long_term_memory".to_string(), JsonValue::String(long_term_memory.clone()));
        }


        // Create the initial context for the chat using the model's prompt template and tokenize it
        let full_prompt = self.model_type.create_chat_prompt(system_prompt, &chat_history, self.extra_data.as_ref());
        let initial_context = self.infer_iter.last_context().model.borrow().tokenize(full_prompt);

        // Reset the InferIter for the chat with the new initial context
        self.infer_iter.reset(initial_context);
        self.chat_history = chat_history;
        self.long_term_memory = long_term_memory;
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
        // If the chat history is too long, we want to compress the memory.
        if self.estimate_token_len() > COMPRESS_MEMORY_THRESHOLD {
            self.compress_memory();
        }

        // Add the beginning of the message prompt to the context
        self.infer_iter
            .push_str(self.model_type.create_chat_message_begin_prompt(sender));

        // If begin_sequence is Some, add it to the message buffer as well
        if let Some(begin_sequence) = begin_sequence {
            self.infer_iter.push_str(begin_sequence);
        }

        // Build end sequences
        let mut full_end_sequences = vec![self.model_type.chat_message_end_sequence()];
        full_end_sequences.extend_from_slice(end_sequences);

        // Then infer the response from the model until we get one of the end sequences back
        let response = self
            .infer_iter
            .complete(&full_end_sequences);

        // Reset the message buffer to just the end message prompt
        self.infer_iter.push_str(self.model_type.create_chat_message_end_prompt());

        // Add the inferred response to the chat history as a new message
        let message = ChatMessage::new(sender.clone(), response.unwrap());
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

    /// Get the entire chat history.
    pub fn history(&self) -> &[ChatMessage] {
        &self.chat_history
    }

    /// Estimate the token length of the current full chat context.
    fn estimate_token_len(&self) -> usize {
        // Start with the InferIter context
        let mut tokens = self.infer_iter.last_context().len();
        // Add an estimate for pending context
        tokens += self.infer_iter.pending_context().len() / ESTIMATE_CHARACTERS_PER_TOKEN;
        
        tokens
    }

    /// Compress the memory of the chat by summarizing the past_memory and half of the chat history together,
    /// storing the summary in past_memory, and then resetting the chat history to just the other half of the chat history.
    fn compress_memory(&mut self) {
        // Start with either the long_term_memory or an empty string as the base for the summary prompt
        let mut prompt = self.long_term_memory.clone().map_or_else(String::new, |mem| format!("{}\n\n", mem));

        // Get the message count of half of the chat history (the older half) to include in the summary
        let half = self.chat_history.len() / 2;
        // If we will end on a user message then decrement by one
        let half = if half > 1 && matches!(self.chat_history[half - 1].sender(), ChatRole::User) {
            half - 1
        } else {
            half
        };

        // Get the last system prompt message appearing in the history before half
        let new_system_prompt = self.chat_history[..half]
            .iter()
            .rev()
            .find(|msg| matches!(msg.sender(), ChatRole::Other(name) if name == "system"))
            .map(|msg| msg.content())
            .unwrap_or(DEFAULT_SYSTEM_PROMPT)
            .to_string();

        // Then add the oldest half of the chat history to the prompt
        for message in &self.chat_history[..half] {
            prompt.push_str(&format!("{}: {}\n", message.sender(), message.content()));
        }

        // Get just the newer half of the chat history as a new vec
        let new_chat_history = self.chat_history[half..].to_vec();

        // Reset self to a new state for summarization
        self.reset(
            "You are an assistant who summarizes conversations between yourself and the user in a concise manner.",
            vec![],
            self.long_term_memory.clone(),
        );

        // Add message to the chat asking for a summary of the prompt to be generated
        self.push_message(ChatMessage::new(
            ChatRole::User,
            format!(
                "Summarize this conversation between us in a concise manner, preserving important details and context:\n{}",
                prompt
            ),
        ));

        // Infer the summary from the model
        let summary = self.infer_message(
            &ChatRole::Assistant,
            Some("Summary:\n"),
            &[],
        ).trim()
         .to_string();

        // Reset self to the new state for further conversing
        self.reset(
            new_system_prompt,
            new_chat_history,
            Some(summary),
        );

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

/// Represents the role of a participant in a chat.
#[derive(Clone, Debug)]
pub enum ChatRole {
    User,
    Assistant,
    Other(String),
}

impl Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
            ChatRole::Other(name) => write!(f, "{}", name),
        }
    }
}
