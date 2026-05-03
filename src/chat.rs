use std::fmt::Display;
use std::rc::Rc;

use anyhow::Result;
use serde_json::json;

use crate::chat_schema::{ChatSchema, input_key, output_key};
use crate::chat_wrapper::{ChatWrapper, SimpleChatWrapper};
use crate::data::{JsonMap, JsonValue, StringMap};
use crate::model::{MAX_TOKENS, Model};
use crate::prelude::{InferIter, InferParams, ModelType, TokenString};
use crate::string_map;

const LONG_CONTEXT_THRESHOLD: usize = MAX_TOKENS * 2 / 3;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParameterType {
    String,
    Enum(Vec<String>),
    Number,
    Boolean,
}

impl ParameterType {
    pub fn string() -> Self {
        ParameterType::String
    }

    pub fn enumeration(options: impl Into<Vec<String>>) -> Self {
        ParameterType::Enum(options.into())
    }

    pub fn number() -> Self {
        ParameterType::Number
    }

    pub fn boolean() -> Self {
        ParameterType::Boolean
    }

    pub fn to_json_type_name(&self) -> &'static str {
        match self {
            ParameterType::String => "string",
            ParameterType::Enum(_) => "string",
            ParameterType::Number => "number",
            ParameterType::Boolean => "boolean",
        }
    }
}

/// A single parameter definition for a tool.
/// Contains the parameter's name, type and description, as well as a default value which also makes the parameter optional if it is Some.
#[derive(Clone, Debug)]
pub struct ParameterDefinition {
    name: String,
    param_type: ParameterType,
    description: String,
    default_value: Option<JsonValue>,
}

impl ParameterDefinition {
    pub fn new(
        name: impl Display,
        description: impl Display,
        param_type: ParameterType,
        default_value: Option<JsonValue>,
    ) -> Self {
        // Verify that if there is a default value then it matches the parameter type
        if let Some(default_value) = &default_value {
            let type_matches = match &param_type {
                ParameterType::String => default_value.is_string(),
                ParameterType::Number => default_value.is_number(),
                ParameterType::Boolean => default_value.is_boolean(),
                ParameterType::Enum(options) => {
                    if let Some(default_str) = default_value.as_str() {
                        options.contains(&default_str.to_string())
                    } else {
                        false
                    }
                }
            };
            if !type_matches {
                panic!("Default value for parameter '{}' does not match its type or allowed enum values", name);
            }
        }

        Self {
            name: name.to_string(),
            param_type,
            description: description.to_string(),
            default_value,
        }
    }
}

/// Defines a tool which can be used by the chat in tool calls.
/// The tool is defined by a name, a description and a function which takes in the tool's arguments as a StringMap and returns a String result.
#[derive(Clone)]
pub struct Tool {
    name: String,
    description: String,
    parameters: Vec<ParameterDefinition>,
    func: Rc<Box<dyn Fn(JsonMap) -> Result<JsonValue>>>,
}

impl Tool {
    /// Creates a new tool definition.
    pub fn new<R: Into<JsonValue> + 'static>(
        name: impl Display,
        description: impl Display,
        parameters: impl IntoIterator<Item = ParameterDefinition>,
        func: impl Fn(JsonMap) -> Result<R> + 'static,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            parameters: parameters.into_iter().collect(),
            func: Rc::new(Box::new(move |args| Ok(func(args)?.into()))), // Wrap the provided function to convert its result into a JsonValue
        }
    }

    /// Gets the name of the tool.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Creates a JSON schema representation of the tool for presenting to the model.
    pub fn to_json_schema(&self) -> JsonValue {
        json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.iter().map(|param| {
                        // Base schema for the parameter with its type and description
                        let mut property_schema = json!({
                            "type": param.param_type.to_json_type_name(),
                            "description": param.description,
                        });
                        // If this is an enum parameter then we also add the allowed enum values to the schema
                        if let ParameterType::Enum(options) = &param.param_type {
                            property_schema["enum"] = json!(options);
                        }
                        (param.name.clone(), property_schema)
                    }).collect::<serde_json::Map<_, _>>(),
                    "required": self.parameters.iter().filter_map(|param| {
                        // Only require parameters which don't have a default value
                        if param.default_value.is_none() {
                            Some(param.name.clone())
                        } else {
                            None
                        }
                    }).collect::<Vec<_>>(),
                },
            },
        })
    }
}

/// Represents a call to a tool from the model, containing the tool definition and the arguments for the tool call.
pub struct ToolCall {
    tool: Tool,
    args: JsonMap,
}

impl ToolCall {
    /// Gets the tool definition for this tool call.
    pub fn tool(&self) -> &Tool {
        &self.tool
    }

    /// Gets the arguments for this tool call.
    pub fn args(&self) -> &JsonMap {
        &self.args
    }

    /// Executes the tool call by calling the tool's function with the provided arguments and returns the result.
    /// Also pushes the result to the chat history as a message from the tool.
    pub fn execute(&self, chat: &mut Chat) -> Result<JsonValue> {
        let result = (self.tool.func)(self.args.clone())?;

        // Push the tool response to the chat history as a message from the tool
        chat.push_message(ChatMessage::new(ChatRole::Tool, serde_json::to_string_pretty(&result)?));

        Ok(result)
    }
}

/// A single inferred message from the model, containing the content of the message as well as any other relevant information.
pub struct InferredMessage {
    content: String,
    tool_call: Option<ToolCall>,
}

impl InferredMessage {
    /// Returns the content of the inferred message.
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the tool call of the inferred message, if any.
    pub fn tool_call(&self) -> Option<&ToolCall> {
        self.tool_call.as_ref()
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

        // Retrieve tool calls from the response (while also removing them from response_result)
        let mut response_result = response_result;
        let mut tool_call = None;
        while let Some(tool_call_start) = response_result.find("<tool_call>") {
            if let Some(tool_call_end) = response_result.find("</tool_call>") {
                let tool_call_str = &response_result[tool_call_start + "<tool_call>".len()..tool_call_end];
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
                            tool_call = Some(ToolCall { tool: tool.clone(), args });
                        } else {
                            println!("Tool '{}' not found", tool_name);
                        }
                    }
                }
                else {
                    println!("Failed to parse tool call JSON: {}", tool_call_str);
                }
                // Remove the tool call from the response result
                response_result.replace_range(tool_call_start..tool_call_end + "</tool_call>".len(), "");
            } else {
                break; // If there is a start tag without an end tag, we stop looking for more tool calls
            }
        }

        // Add the inferred response to the chat history as a new message
        let message = ChatMessage::new(sender.clone(), response_result.clone());
        self.chat_history.push(message);
        Some((
            InferredMessage {
                content: response_result,
                tool_call,
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
        self.token_len() > LONG_CONTEXT_THRESHOLD
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
            &InferParams::new_logical(),
            system_schema,
            input_schema,
            output_schema,
            &example_pairs,
            [
                "Summarize the provided conversation in a concise way, while retaining as much **useful** information as possible.".to_string(),
                "The summary should be wrapped in '\"' and as short as possible (preferably one paragraph) while being informative.".to_string(),
                "\"assistant\" and \"user\" in the conversation should be referred to as \"you\" and \"the user\" respectively in the summary.".to_string(),
            ],
            []
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
            "Chat compressed. From {} tokens to {} tokens.",
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