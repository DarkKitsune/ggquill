use std::fmt::Display;

use crate::data::StringMap;
use crate::model::{MAX_TOKENS, Model};
use crate::prelude::{InferIter, InferParams, ModelType, TokenString};

const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant that tries to help the user with their requests as best as you can. \
    You should always try to be as helpful and informative as possible, while also being concise and clear in your responses. \
    Avoid unnecessary repetition and fluff, and try to get to the point quickly.";
const COMPRESS_MEMORY_THRESHOLD: usize = MAX_TOKENS / 2;

/// Represents a chat between user and model.
pub struct Chat {
    system_prompt: String,
    model_type: ModelType,
    infer_iter: InferIter,
    /// The messages in the chat. This should be synchronized with the context of the infer_iter.
    chat_history: Vec<ChatMessage>,
    /// When there are too many messages we summarize the long_term_memory and half of the chat history together,
    /// store the summary in long_term_memory.
    long_term_memory: Option<String>,
    how_to_respond: Vec<String>,
    extra_data: Option<StringMap>,
}

impl Chat {
    /// Creates a new chat.
    pub fn new(
        model: &mut Model,
        system_prompt: impl AsRef<str>,
        chat_history: &[ChatMessage],
        infer_params: &InferParams,
        how_to_respond: impl Into<Vec<String>>,
        extra_data: Option<StringMap>,
    ) -> Self {
        let how_to_respond = how_to_respond.into();
        // Create the initial context for the chat using the model's prompt template and tokenize it
        let full_prompt = model.model_type().create_chat_prompt(
            &system_prompt,
            chat_history,
            &how_to_respond
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            extra_data.as_ref(),
        );
        let initial_context = model.tokenize(full_prompt);

        // Create the InferIter for the chat with the initial context
        let infer_iter = model.infer_iter(initial_context, infer_params).unwrap();

        Self {
            system_prompt: system_prompt.as_ref().to_string(),
            model_type: model.model_type(),
            infer_iter,
            chat_history: chat_history.to_vec(),
            long_term_memory: None,
            how_to_respond,
            extra_data,
        }
    }

    /// Resets the chat using a given parameters.
    pub fn reset(
        &mut self,
        system_prompt: impl AsRef<str>,
        chat_history: Vec<ChatMessage>,
        long_term_memory: Option<String>,
    ) {
        self.system_prompt = system_prompt.as_ref().to_string();
        // If long term memory is provided, set it in key "memory" of self.extra_data for the prompt template
        // Also create self.extra_data if it doesn't exist yet
        if let Some(long_term_memory) = &long_term_memory {
            let extra_data = self.extra_data.get_or_insert_with(StringMap::new);
            extra_data.insert("Your past memory".to_string(), long_term_memory.clone());
        }

        // Create the initial context for the chat using the model's prompt template and tokenize it
        let full_prompt = self.model_type.create_chat_prompt(
            &self.system_prompt,
            &chat_history,
            &self
                .how_to_respond
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            self.extra_data.as_ref(),
        );
        let initial_context = self
            .infer_iter
            .last_context()
            .model
            .borrow()
            .tokenize(full_prompt);

        // Reset the InferIter for the chat with the new initial context
        self.infer_iter.reset(initial_context);
        self.chat_history = chat_history;
        self.long_term_memory = long_term_memory;
    }

    /// Gets the chat's current state (system prompt, chat history, and long term memory) as a tuple.
    pub fn get_state(&self) -> (String, Vec<ChatMessage>, Option<String>) {
        (
            self.system_prompt.clone(),
            self.chat_history.clone(),
            self.long_term_memory.clone(),
        )
    }

    /// Push an existing message to the chat history.
    pub fn push_message(&mut self, message: ChatMessage) {
        // First add the complete message prompt to the message buffer
        self.infer_iter.push_str(
            self.model_type
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
    /// Returns the content of the inferred message and a mutable reference to the InferIter, allowing for further inference.
    pub fn infer_message(
        &mut self,
        sender: &ChatRole,
        begin_sequence: Option<&str>,
        end_sequences: &[&str],
    ) -> &str {
        self.infer_message_ext(sender, begin_sequence, end_sequences, |_, _, _| Some(()))
            .unwrap()
            .0
    }

    /// Infer a new message from the model and push it to the chat history.
    /// If a begin_sequence is provided, it is treated as if it was prepended to the model's response, influencing the inference.
    /// Returns the content of the inferred message and a mutable reference to the InferIter, allowing for further inference.
    /// The after_response callback is given the message, a mutable reference to the InferIter and the end sequence which caused the
    /// inference to end, allowing for further inference to be done before ending the message.
    pub fn infer_message_ext<R>(
        &mut self,
        sender: &ChatRole,
        begin_sequence: Option<&str>,
        end_sequences: &[&str],
        // Callback which is given the message, a mutable reference to the internal InferIter after the inference is done,
        // as well as the end sequence (if any) which caused the inference to end,
        // allowing for further inference to be done before ending the message.
        // If the callback returns None then the message will end immediately and not be added to the chat history
        mut after_response: impl FnMut(&str, &mut InferIter, Option<&str>) -> Option<R>,
    ) -> Option<(&str, R)> {
        // If the chat history is too long, we want to compress the memory.
        if self.token_len() > COMPRESS_MEMORY_THRESHOLD {
            self.compress_memory();
        }

        // Store the current InferIter context before the message
        let context_before_message = self.infer_iter.full_context();

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
        let response = self.infer_iter.complete(&full_end_sequences);

        // Get the result of the inference, with any U+FFFD replacement characters removed (indicates invalid UTF-8)
        let response_result = response.result().replace('\u{FFFD}', "").trim().to_string();

        // If there is an after_response callback then call it with a mutable reference to the InferIter and the end sequence which caused the inference to end
        let after_response_result = after_response(
            &response_result,
            &mut self.infer_iter,
            response.end_sequence(),
        );

        // If the after_response callback returned None then we reset self.infer_iter to context_before_message and return None
        let after_response_result = match after_response_result {
            Some(result) => result,
            None => {
                self.infer_iter.reset(context_before_message);
                return None;
            }
        };

        // Reset the message buffer to just the end message prompt
        self.infer_iter
            .push_str(self.model_type.create_chat_message_end_prompt());

        // Add the inferred response to the chat history as a new message
        let message = ChatMessage::new(sender.clone(), response_result);
        self.chat_history.push(message);
        Some((
            self.chat_history.last().unwrap().content(),
            after_response_result,
        ))
    }

    /// Add a message more directly by a callback which uses the underlying InferIter to write the message body
    /// and returns the message content as a string.
    /// The message written to the InferIter should match the message string returned by the callback, but this is not enforced.
    pub(crate) fn add_message_with_infer_iter<R: AsRef<str>>(
        &mut self,
        sender: &ChatRole,
        infer_func: impl FnOnce(&mut InferIter) -> R,
    ) -> R {
        // Start message
        self.infer_iter
            .push_str(self.model_type.create_chat_message_begin_prompt(sender));

        // Use callback for message body
        let returned = infer_func(&mut self.infer_iter);

        // End message
        self.infer_iter.push_str("\n");
        self.infer_iter
            .push_str(self.model_type.create_chat_message_end_prompt());

        // Add the message to the chat history
        let message = ChatMessage::new(sender.clone(), returned.as_ref().to_string());
        self.chat_history.push(message);

        returned
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

    /// Get the token length of the current full chat context.
    fn token_len(&self) -> usize {
        self.infer_iter.last_context().len()
            + self
                .infer_iter
                .pending_context()
                .map_or(0, |pending| pending.len())
    }

    /// Compress the memory of the chat by summarizing the past_memory and half of the chat history together,
    /// storing the summary in past_memory, and then resetting the chat history to just the other half of the chat history.
    fn compress_memory(&mut self) {
        println!(
            "Compressing memory... Current token length: {}",
            self.token_len()
        );

        // Start with either the long_term_memory or an empty string as the base for the summary prompt
        let mut prompt = self
            .long_term_memory
            .clone()
            .map_or_else(String::new, |mem| format!("{}\n\n", mem));

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
        let summary = self
            .infer_message(&ChatRole::Assistant, Some("Summary:\n"), &[])
            .trim()
            .to_string();

        // Reset self to the new state for further conversing
        self.reset(new_system_prompt, new_chat_history, Some(summary));

        println!("Memory compressed. New token length: {}", self.token_len());
    }

    /// Update the inference parameters for the chat's InferIter.
    pub fn update_infer_params(&mut self, params: &InferParams) {
        self.infer_iter.update_params(params);
    }

    /// Get all tokens that make up the chat's context so far.
    pub fn get_tokens(&self) -> TokenString {
        self.infer_iter.full_context()
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
        Self {
            sender,
            content: content.to_string(),
        }
    }

    /// Creates a new chat message from the assistant.
    pub fn assistant(content: impl Display) -> Self {
        Self::new(ChatRole::Assistant, content)
    }

    /// Creates a new chat message from the user.
    pub fn user(content: impl Display) -> Self {
        Self::new(ChatRole::User, content)
    }

    /// Creates a new chat message from the system.
    pub fn system(content: impl Display) -> Self {
        Self::new(ChatRole::System, content)
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
        write!(f, "{}:\n{}", self.sender, self.content)
    }
}

/// Represents the role of a participant in a chat.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    System,
    Other(String),
}

impl Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
            ChatRole::System => write!(f, "system"),
            ChatRole::Other(name) => write!(f, "{}", name),
        }
    }
}
