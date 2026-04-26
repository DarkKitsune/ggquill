use crate::prelude::*;

/// Builds JSON objects using the given instructions
pub struct JsonBuilder {
    chat_wrapper: SimpleChatWrapper,
}

impl JsonBuilder {
    /// Creates a new JsonBuilder with the provided model. The model should be a capable instruction-following model.
    pub fn new(model: &mut Model) -> Self {
        let system_schema = "You are a helpful assistant that builds JSON objects based on the provided instructions. \
            Follow the instructions carefully to construct the JSON object. \
            Ensure that the JSON is well-formed and only includes relevant fields based on the instructions.";
        let input_schema =
            ChatSchema::new().with_text(Some("Instructions".to_string()), "<instructions>");
        let output_schema = ChatSchema::new().with_json(Some("JSON".to_string()), "json");

        // Example pairs of input instructions and expected output JSON for the chat wrapper
        let examples = [
            (
                string_map! {
                    "instructions" => "Build a JSON object for a product with the name 'Laptop', the property 'price' set to 999.99, and a boolean property 'in_stock' which is enabled.",
                },
                string_map! {
                    "json" => r#"{"name": "Laptop", "price": 999.99, "in_stock": true}"#,
                },
            ),
            (
                string_map! {
                    "instructions" => "Construct a JSON object for a fantasy romance book with an interesting title, an author name starting with 'J', and a set of characters including a brave hero and a mysterious love interest.",
                },
                string_map! {
                    "json" =>
r#"{
    "title": "One Night With a Fire Elemental",
    "author": "Jasmine Silver",
    "characters": [
        {
            "name": "Ragnar The Stout",
            "description": "Brave Hero"
        },
        {
            "name": "Luna",
            "description": "Mysterious Love Interest"
        }
    ]
}"#,
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
        );

        Self { chat_wrapper }
    }

    /// Builds a JSON object based on the provided instructions and returns it as a string.
    /// Returns None if a valid JSON object could not be generated after the given number of attempts.
    /// If `max_attempts` is None then it will keep trying indefinitely until a valid JSON is generated.
    pub fn build_json(
        &mut self,
        instructions: &str,
        max_attempts: Option<usize>,
    ) -> Option<JsonValue> {
        // Create the input context for the chat wrapper using the provided instructions
        let input_context = string_map! {
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
            if let Ok(parsed_json) = serde_json::from_str::<JsonValue>(&captures["json"]) {
                break Some(parsed_json);
            } else {
                // If we fail to parse the JSON, reset the chat wrapper to the saved state
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
}
