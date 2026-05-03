use std::fmt::Display;

use crate::prelude::*;

/// Formats and humanizes a list of input values into a single well-formatted string using a model.
pub struct Humanizer {
    chat_wrapper: SimpleChatWrapper,
}

impl Humanizer {
    /// Creates a new Humanizer with the provided model, seed, and temperature.
    pub fn new(model: Model) -> Self {
        let system_schema = "You are a helpful writing assistant. \
        Your goal is to take a list of input strings or values separated by '|' and join them together into a string that a person can easily read and understand. \
        The final output should be a well formatted and grammatically correct string in double quotes which incorporates all of the inputs. \
        If the input values contain JSON, you should extract relevant information from the JSON and incorporate it into the final output string in a natural way.";
        let input_schema = ChatSchema::new().with_text(Some("Inputs".to_string()), "<values>");
        let output_schema =
            ChatSchema::new().with_text(Some("Joined".to_string()), "\"<\" :: joined>");

        // Example pairs of input contexts and expected output values for the chat wrapper
        let examples = [
            (
                string_map! {
                    "values" => "an apple|orange|banana",
                },
                string_map! {
                    "joined" => "an apple, an orange, and a banana",
                },
            ),
            (
                string_map! {
                    "values" => "Please inform | Jack | meeting | 3PM | !",
                },
                string_map! {
                    "joined" => "Please inform Jack about the meeting at 3:00 PM!",
                },
            ),
            (
                string_map! {
                    "values" => "I am | drive | red | big | car | Mira's House.",
                },
                string_map! {
                    "joined" => "I am driving a big red car to Mira's House.",
                },
            ),
            // To support labelled values
            (
                string_map! {
                    "values" => "Weather today|sunny 75% | cloudy 25%| high: 75 degrees |low: 55 degrees",
                },
                string_map! {
                    "joined" => "The weather today is mostly sunny with a high of 75 degrees and a low of 55 degrees.",
                },
            ),
            // To support JSON being joined with strings
            (
                string_map! {
                    "values" => "There was a | {\"name\": \"dog\", \"color\": \"brown\", \"size\": \"big\"} | in the park.",
                },
                string_map! {
                    "joined" => "There was a big brown dog in the park.",
                },
            ),
        ];

        // Create the chat wrapper with the model, schemas, and examples
        let chat_wrapper = SimpleChatWrapper::new(
            model,
            &InferParams::new_logical(),
            system_schema,
            input_schema,
            output_schema,
            &examples,
            vec![],
        )
        .0;

        Self { chat_wrapper }
    }

    /// Joins the provided values into a single string using the model. The input values will be separated by '|'.
    pub fn join(&mut self, values: &[impl Display]) -> String {
        // Format the input values into a single string separated by '|'
        let input_values = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" | ");

        // Create the input context for the chat wrapper
        let input_context = string_map! {
            "values" => input_values,
        };

        // Get the output from the chat wrapper using the input context
        self.chat_wrapper
            .get_output(&input_context)
            .into_captures()
            .remove("joined")
            .unwrap()
    }
}
