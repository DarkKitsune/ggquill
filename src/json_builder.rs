use std::{collections::HashMap, fmt::Display};

use anyhow::Result;

use crate::prelude::*;

/// Represents a node in a JSON template, which can be a primitive type (string, number, boolean), an array, or an object with properties.
#[derive(Debug, Clone)]
pub enum TemplateNode {
    String,
    Number(Option<f64>, Option<f64>), // Optional min and max bounds for numbers
    Boolean,
    Array(Box<TemplateNode>), // Array of a certain type
    Object(Vec<TemplateProperty>), // Object with properties
}

impl TemplateNode {
    pub fn string() -> Self {
        TemplateNode::String
    }

    pub fn number(min: Option<f64>, max: Option<f64>) -> Self {
        TemplateNode::Number(min, max)
    }

    pub fn boolean() -> Self {
        TemplateNode::Boolean
    }

    pub fn array(node: TemplateNode) -> Self {
        TemplateNode::Array(Box::new(node))
    }

    pub fn object(properties: impl Into<Vec<TemplateProperty>>) -> Self {
        TemplateNode::Object(properties.into())
    }

    /// Returns a Json object schema representation of this template node, which can be used for validation or documentation purposes.
    pub fn to_json_schema(&self) -> JsonValue {
        match self {
            TemplateNode::String => json!({"type": "string"}),
            TemplateNode::Number(min, max) => {
                let mut schema = json!({"type": "number"});
                if let Some(min) = min {
                    schema["minimum"] = json!(min);
                }
                if let Some(max) = max {
                    schema["maximum"] = json!(max);
                }
                schema
            }
            TemplateNode::Boolean => json!({"type": "boolean"}),
            TemplateNode::Array(item_node) => {
                json!({
                    "type": "array",
                    "items_schema": item_node.to_json_schema(),
                })
            }
            TemplateNode::Object(properties) => {
                let properties_schema: HashMap<_, _> = properties
                    .iter()
                    .map(|prop| (prop.name.clone(), prop.node.to_json_schema()))
                    .collect();
                let required_fields: Vec<_> = properties
                    .iter()
                    .filter(|prop| prop.required)
                    .map(|prop| prop.name.clone())
                    .collect();
                json!({
                    "type": "object",
                    "properties": properties_schema,
                    "required": required_fields,
                })
            }
        }
    }

    /// Validates a given JSON value against this template node, returning an error if the value does not conform to the template.
    pub fn validate(&self, value: &JsonValue) -> Result<()> {
        match self {
            TemplateNode::String => {
                if !value.is_string() {
                    anyhow::bail!("Expected a string value, but got: {}", value);
                }
            }

            TemplateNode::Number(min, max) => {
                if let Some(num) = value.as_f64() {
                    if let Some(min) = min && num < *min {
                        anyhow::bail!("Number {} is less than minimum allowed value {}", num, min);
                    }
                    if let Some(max) = max && num > *max {
                        anyhow::bail!("Number {} is greater than maximum allowed value {}", num, max);
                    }
                } else {
                    anyhow::bail!("Expected a number value, but got: {}", value);
                }
            }

            TemplateNode::Boolean => {
                if !value.is_boolean() {
                    anyhow::bail!("Expected a boolean value, but got: {}", value);
                }
            }

            TemplateNode::Array(item_node) => {
                if let Some(arr) = value.as_array() {
                    for (index, item) in arr.iter().enumerate() {
                        item_node.validate(item).map_err(|e| {
                            anyhow::anyhow!("Array item at index {} is invalid: {}", index, e)
                        })?;
                    }
                } else {
                    anyhow::bail!("Expected an array value, but got: {}", value);
                }
            }

            TemplateNode::Object(properties) => {
                if let Some(obj) = value.as_object() {
                    // Check if all required properties are present, and validate all properties that are present
                    for prop in properties {
                        // If property is one of the reserved names, we should reject it to avoid confusion with the JSON schema representation
                        match prop.name.as_str() {
                            "type" | "properties" | "items_schema" | "required" => {
                                anyhow::bail!("Property name '{}' is reserved and cannot be used in the template", prop.name);
                            }
                            _ => {}
                        }
                        if let Some(prop_value) = obj.get(&prop.name) {
                            prop.node.validate(prop_value).map_err(|e| {
                                anyhow::anyhow!("Property '{}' is invalid: {}", prop.name, e)
                            })?;
                        } else if prop.required {
                            anyhow::bail!("Missing required property: '{}'", prop.name);
                        }
                    }
                } else {
                    anyhow::bail!("Expected an object value, but got: {}", value);
                }
            }
        }

        Ok(())
    }
}

impl Display for TemplateNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // For simplicity we convert this to a json schema JsonValue and then display that
        let json_schema = self.to_json_schema();
        write!(f, "{}", serde_json::to_string_pretty(&json_schema).unwrap())
    }
}

/// Represents a property in a JSON object template, which can be required or optional and has a name and a type defined by a TemplateNode.
#[derive(Debug, Clone)]
pub struct TemplateProperty {
    pub name: String,
    pub node: TemplateNode,
    pub required: bool,
}

impl TemplateProperty {
    pub fn required(name: &str, node: TemplateNode) -> Self {
        TemplateProperty {
            name: name.to_string(),
            node,
            required: true,
        }
    }

    pub fn optional(name: &str, node: TemplateNode) -> Self {
        TemplateProperty {
            name: name.to_string(),
            node,
            required: false,
        }
    }
}

/// Helper function for creating a TemplateNode object with a more concise syntax.
pub fn object(properties: impl Into<Vec<TemplateProperty>>) -> TemplateNode {
    TemplateNode::object(properties)
}

/// Helper function for creating a TemplateNode array with a more concise syntax.
pub fn array(item_node: TemplateNode) -> TemplateNode {
    TemplateNode::array(item_node)
}

/// Helper function for creating a TemplateNode string with a more concise syntax.
pub fn string() -> TemplateNode {
    TemplateNode::string()
}

/// Helper function for creating a TemplateNode number with a more concise syntax.
pub fn number(min: Option<f64>, max: Option<f64>) -> TemplateNode {
    TemplateNode::number(min, max)
}

/// Helper function for creating a TemplateNode boolean with a more concise syntax.
pub fn boolean() -> TemplateNode {
    TemplateNode::boolean()
}

/// Helper function for creating a required TemplateProperty with a more concise syntax.
pub fn property(name: &str, node: TemplateNode) -> TemplateProperty {
    TemplateProperty::required(name, node)
}

/// Helper function for creating an optional TemplateProperty with a more concise syntax.
pub fn optional_property(name: &str, node: TemplateNode) -> TemplateProperty {
    TemplateProperty::optional(name, node)
}

/// Builds JSON objects using the given instructions
pub struct JsonBuilder {
    chat_wrapper: SimpleChatWrapper,
}

impl JsonBuilder {
    /// Creates a new JsonBuilder with the provided model. The model should be a capable instruction-following model.
    pub fn new(model: &mut Model) -> Self {
        let system_schema = "You are a helpful assistant that builds JSON objects based on the provided JSON schema and instructions. \
            Follow the instructions carefully to construct the JSON object.";
        // Input schema defines the structure of the user instructions, which is just a labelled text block containing the instructions
        let input_schema =
            ChatSchema::new()
                .with_text(Some("JSON Schema".to_string()), "<template>")
                .with_text(Some("Instructions".to_string()), "<instructions>");
        // Output schema defines the structure of the assistant response, which is a labelled JSON block in this case
        let output_schema = ChatSchema::new().with_json(Some("JSON".to_string()), "json");

        // Example pairs of input instructions and expected output JSON for the chat wrapper
        let examples = [
            (
                string_map! {
                    "template" => object(
                        [
                            property("title", string()),
                            optional_property("year", number(Some(1900.0), None)),
                            property("main_actors", array(object(
                                [
                                    property("name", string()),
                                    optional_property("tidbits", array(string())),
                                ]
                            ))),
                        ]
                    ),
                    "instructions" => "Build a short and simple JSON for the movie 'Inception' with just 3 actors.",
                },
                string_map! {
                    "json" =>
r#"{
    "title": "Inception",
    "year": 2010,
    "main_actors": [
        {
            "name": "Leonardo DiCaprio",
            "tidbits": ["Played the role of Dom Cobb, the main protagonist.", "Nominated for an Academy Award for this role."]
        },
        {
            "name": "Joseph Gordon-Levitt",
            "tidbits": ["Played the role of Arthur, Dom Cobb's partner.", "Performed many of his own stunts in the film."]
        },
        {
            "name": "Elliot Page",
        }
    ]
}"
"#,
                },
            ),
        ];

        // Create the chat wrapper with the specified schemas and examples
        let chat_wrapper = SimpleChatWrapper::new(
            model,
            &InferParams::new_logical(),
            system_schema,
            input_schema,
            output_schema,
            &examples,
            vec![
                "Ensure that the JSON is well-formed and only includes relevant fields based on the instructions.".to_string(),
            ],
        );

        Self { chat_wrapper }
    }

    /// Builds a JSON object based on the provided instructions and returns it as a string.
    /// Returns None if a valid JSON object could not be generated after the given number of attempts.
    /// If `max_attempts` is None then it will keep trying indefinitely until a valid JSON is generated.
    pub fn build_json(
        &mut self,
        instructions: &str,
        template: &TemplateNode,
        max_attempts: Option<usize>,
    ) -> Option<JsonValue> {
        // Create the input context for the chat wrapper using the provided instructions
        let input_context = string_map! {
            "template" => template,
            "instructions" => instructions,
        };

        // Save the chat wrapper state in case we need to retry generating the output JSON
        let saved_state = self.chat_wrapper.get_state();

        // Loop to keep trying to generate a valid JSON output until we succeed or reach the maximum number of attempts
        let mut attempts = 0;
        loop {
            let json_output = self.chat_wrapper.get_output(&input_context);
            let captures = json_output.captures();

            // Try parsing the JSON string to ensure it's well-formed, and return it as a string
            println!("Attempting to build JSON (attempt {})", attempts + 1);
            if let Ok(parsed_json) = serde_json::from_str::<JsonValue>(&captures["json"])
            {
                // Run template validation on the JSON object
                match template.validate(&parsed_json) {
                    Ok(_) => {
                        break Some(parsed_json);
                    }
                    Err(e) => {
                        println!("Generated JSON is invalid according to the template: {}", e);
                    }
                }
            }

            // We failed to parse the JSON, so reset the chat wrapper to the saved state
            self.chat_wrapper.reset(&saved_state);

            // Then increment the attempt counter and check if we've reached the maximum number of attempts (if specified)
            attempts += 1;
            if let Some(max_attempts) = max_attempts
                && attempts >= max_attempts
            {
                break None;
            }
        }
    }
}
