use std::fmt::Display;

use crate::prelude::*;

/// A saved checkpoint in the chat, which can be used to reset the chat to a previous state.
/// This currently only saves text data and the context of the internal `InferIter`.
pub struct ChatCheckpoint {
    long_term_memory: Option<String>,
    chat_history: Vec<ChatMessage>,
    context_tokens: TokenString,
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
    Tool,
    Other(String),
}

impl Display for ChatRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatRole::User => write!(f, "user"),
            ChatRole::Assistant => write!(f, "assistant"),
            ChatRole::System => write!(f, "system"),
            ChatRole::Tool => write!(f, "tool"),
            ChatRole::Other(name) => write!(f, "{}", name),
        }
    }
}

/// A single inferred message from the model, containing the content of the message as well as any other relevant information.
pub struct InferredMessage {
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<ToolCall>,
    malformed_tool_call: bool,
}

impl InferredMessage {
    /// Returns the content of the inferred message.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the tool call of the inferred message, if any.
    pub fn tool_calls(&self) -> &[ToolCall] {
        &self.tool_calls
    }

    /// Returns the reasoning behind the inferred message, if any.
    pub fn reasoning(&self) -> Option<&str> {
        self.reasoning.as_deref()
    }

    /// Returns whether the inferred message contained a malformed tool call (i.e. it had a <tool_call> tag but we failed to parse the JSON inside it).
    pub fn malformed_tool_call(&self) -> bool {
        self.malformed_tool_call
    }
}

/// Represents a chat between user and model.
pub struct Chat {
    system_prompt: String,
    model_type: ModelType,
    infer_iter: InferIter,
    /// The messages in the chat. This should be synchronized with the context of the infer_iter.
    chat_history: Vec<ChatMessage>,
    /// When there are too many messages we summarize the long_term_memory and half of the chat history together,
    /// store the summary in long_term_memory.
    /// This should not be written to unless you know what you are doing, as it's meant to be synchronized with the infer_iter.
    long_term_memory: Option<String>,
    how_to_respond: Vec<String>,
    extra_data: Option<StringMap>,
    tools: Vec<Tool>,
}

impl Chat {
    /// Creates a new chat.
    pub fn new(
        model: Model,
        system_prompt: impl AsRef<str>,
        chat_history: &[ChatMessage],
        infer_params: &InferParams,
        how_to_respond: impl Into<Vec<String>>,
        extra_data: Option<StringMap>,
        tools: impl Into<Vec<Tool>>,
    ) -> (Self, ChatCheckpoint) {
        let how_to_respond = how_to_respond.into();
        let tools = tools.into();

        // Create the initial context for the chat using the model's prompt template and tokenize it
        let full_prompt = model.model_type().create_chat_prompt(
            &system_prompt,
            chat_history,
            &how_to_respond
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            extra_data.as_ref(),
            &tools,
        );
        let initial_context = model.tokenize(full_prompt);

        // Create the InferIter for the chat with the initial context
        let model_type = model.model_type();
        let infer_iter = model.infer_iter(initial_context, infer_params).unwrap();

        let chat = Self {
            system_prompt: system_prompt.as_ref().to_string(),
            model_type,
            infer_iter,
            chat_history: chat_history.to_vec(),
            long_term_memory: None,
            how_to_respond,
            extra_data,
            tools,
        };

        let checkpoint = chat.create_checkpoint();
        (chat, checkpoint)
    }

    /// Saves the current state of the chat as a ChatCheckpoint which can be used to reset the chat back to this state.
    pub fn create_checkpoint(&self) -> ChatCheckpoint {
        ChatCheckpoint {
            long_term_memory: self.long_term_memory.clone(),
            chat_history: self.chat_history.clone(),
            context_tokens: self.infer_iter.full_context(),
        }
    }

    /// Resets the chat to a previous state captured by `create_checkpoint()`.
    pub fn reset_to_checkpoint(&mut self, checkpoint: &ChatCheckpoint) {
        self.chat_history = checkpoint.chat_history.clone();
        self.long_term_memory = checkpoint.long_term_memory.clone();
        self.infer_iter.reset(checkpoint.context_tokens.clone());
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
    ) -> InferredMessage {
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
    ) -> Option<(InferredMessage, R)> {
        // If the chat history is too long, we want to compress the memory.
        if self.is_context_long() {
            self.compress();
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

        // Push the end message prompt
        self.infer_iter
            .push_str(self.model_type.create_chat_message_end_prompt());

        // Retrieve tool call(s) from the response (while also removing them from response_result)
        let mut response_result = response_result;
        let mut tool_calls = Vec::new();
        let mut found_tool_call = false; // If this is true but tool_call is still None after the loop, then there was a problem parsing the tool call JSON
        while let Some(tool_call_start) = response_result.find("<tool_call>") {
            found_tool_call = true;
            if let Some(tool_call_end) = response_result.find("</tool_call>") {
                let tool_call_str =
                    &response_result[tool_call_start + "<tool_call>".len()..tool_call_end];
                // Parse the tool call JSON
                if let Ok(tool_call_json) = serde_json::from_str::<JsonValue>(tool_call_str) {
                    fn parse_tool_call(tool_call_json: &JsonValue) -> Option<(&str, JsonMap)> {
                        let tool_name = tool_call_json.get("tool")?.as_str()?;
                        let args = tool_call_json.get("args")?.as_object()?.clone();
                        Some((tool_name, args))
                    }

                    if let Some((tool_name, args)) = parse_tool_call(&tool_call_json) {
                        // Find the tool with the given name and pass up a ToolCall to it
                        if let Some(tool) = self.tools.iter().find(|t| t.name() == tool_name) {
                            tool_calls.push(ToolCall::new(tool.clone(), args));
                        } else {
                            println!("Tool '{}' not found", tool_name);
                        }
                    }
                } else {
                    println!("Failed to parse tool call JSON: {}", tool_call_str);
                }
                // Remove the tool call from the response result
                response_result
                    .replace_range(tool_call_start..tool_call_end + "</tool_call>".len(), "");
            } else {
                println!("Found <tool_call> tag but no </tool_call> tag in response");
                // No ending tag so just remove the starting tag and everything after it, since it's probably incomplete if we haven't gotten the end tag yet
                response_result.replace_range(tool_call_start.., "");
                break;
            }
        }
        let malformed_tool_call = found_tool_call && tool_calls.is_empty();

        // Add the inferred response to the chat history as a new message
        let message = ChatMessage::new(sender.clone(), response_result.clone());
        self.chat_history.push(message);
        Some((
            InferredMessage {
                content: response_result,
                reasoning: None, // TODO: add support for reasoning/thinking
                tool_calls,
                malformed_tool_call,
            },
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
    pub fn token_len(&self) -> usize {
        self.infer_iter.last_context().len()
            + self
                .infer_iter
                .pending_context()
                .map_or(0, |pending| pending.len())
    }

    /// Gets whether the chat's context is getting long enough that old details may soon start to be forgotten.
    pub fn is_context_long(&self) -> bool {
        self.token_len() > self.infer_iter.target_context_window()
    }

    /// Compress the chat by joining the long term memory (if it exists) and the oldest half of the chat history into a single string,
    /// summarizing it using the model, creating a new context and resetting the infer_iter to it. Also replaces the old long term memory with the new summary.
    pub(crate) fn compress(&mut self) {
        let old_token_len = self.token_len();

        // Get the oldest half of the chat history and join it into a single string
        let half_history_len = self.chat_history.len() / 2;
        let remaining_history = self.chat_history.split_off(half_history_len);
        let history_string = self
            .chat_history
            .iter()
            .map(ChatMessage::to_string)
            .collect::<Vec<_>>()
            .join("\n\n");

        // Create the full memory string to summarize by joining the long term memory (if it exists) and the history string
        let mut full_memory = self.long_term_memory.clone().unwrap_or_default();
        if !full_memory.is_empty() && !history_string.is_empty() {
            full_memory.push_str("\n\n");
        }
        full_memory.push_str(&history_string);

        // Summarize the full memory string using a ChatWrapper.
        // We create a ChatWrapper on the fly here so we don't need to store it, because compression should be a relatively rare event anyway
        let model = self.infer_iter.clone_model();
        let system_schema = "You are a helpful assistant that summarizes text.";
        let input_schema =
            ChatSchema::new().with_text(Some("Text to Summarize".to_string()), input_key("input"));
        let output_schema = ChatSchema::new().with_text(
            Some("Summary".to_string()),
            output_key("summary", Some("\""), &["\""]),
        );
        let example_pairs = [(
            // Full conversation text
            string_map! {
                "input" =>
"You (the assistant) and the user are talking about how you like cats.

Assistant:
I really like cats, they are so cute and fluffy!

User:
Yeah, I agree! Do you have any cats?

Assistant:
i don't have any cats, but I wish I did! I just love them so much! Do you have any cats?

User:
I have one cat, his name is Whiskers. He's a gray tabby and he's very playful."
            },
            // Summary
            string_map! {
                "summary" => "You and the user are talking about how much you like cats. \
                You express a desire to own cats, and the user has just mentioned owning a playful gray tabby named Whiskers."
            },
        )];
        let mut summarizer = SimpleChatWrapper::new(
            model,
            &InferParams::new_logical().with_memory_priority(MemoryPriority::Low),
            system_schema,
            input_schema,
            output_schema,
            &example_pairs,
            [
                "Summarize the provided conversation in a concise way, while retaining as much **useful** information as possible.".to_string(),
                "The summary should be wrapped in '\"' and as short as possible (preferably one paragraph) while being informative.".to_string(),
                "\"assistant\" and \"user\" in the conversation should be referred to as \"you\" and \"the user\" respectively in the summary.".to_string(),
            ],
        ).0;
        let summary = summarizer
            .get_output(&string_map! {
                "input" => full_memory
            })
            .into_capture("summary")
            .unwrap();

        // Join the system prompt and the new summary together to create a new system prompt with the future long term memory in the context
        let system_prompt_and_summary = format!(
            "{}\n---\n# Conversation So Far (Summarized)\n{}",
            self.system_prompt, summary
        );

        // Create a new context for the chat using the model's prompt template with the remaining half of the chat history, then reset the infer_iter to it.
        let full_prompt = self.model_type.create_chat_prompt(
            system_prompt_and_summary,
            &remaining_history,
            &self
                .how_to_respond
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>(),
            self.extra_data.as_ref(),
            &self.tools,
        );
        let new_context = self.infer_iter.tokenize(full_prompt);
        self.infer_iter.reset(new_context);

        // Update self
        self.long_term_memory = Some(summary.clone());
        self.chat_history = remaining_history;

        println!(
            "\n(Chat compressed. From {} tokens to {} tokens.)\n",
            old_token_len,
            self.token_len()
        );
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
