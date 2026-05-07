use std::{fmt::Display, rc::Rc};

use crate::prelude::*;

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
                panic!(
                    "Default value for parameter '{}' does not match its type or allowed enum values",
                    name
                );
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
    /// Also pushes the result to the chat history as a message from the tool and then returns the content of the next assistant response.
    pub fn execute(&self, chat: &mut Chat) -> Result<(JsonValue, InferredMessage)> {
        let result = (self.tool.func)(self.args.clone())?;

        // Push the tool response to the chat history as a message from the tool
        chat.push_message(ChatMessage::new(
            ChatRole::Tool,
            serde_json::to_string_pretty(&result)?,
        ));

        // Infer the next message from the assistant after the tool call, which may contain the assistant's response to the tool call
        let assistant_response = chat.infer_message(&ChatRole::Assistant, None, &[]);

        Ok((result, assistant_response))
    }

    pub(crate) fn new(tool: Tool, args: Map<String, serde_json::Value>) -> Self {
        Self { tool, args }
    }
}
