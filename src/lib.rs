pub mod actor;
pub mod chat;
pub mod chat_schema;
pub mod chat_wrapper;
pub mod data;
pub mod humanizer;
pub mod inference;
pub mod json_builder;
pub mod model;
pub mod model_type;
pub mod prelude;
pub mod token_string;

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, io::Write};

    use anyhow::Result;

    use crate::prelude::*;

    #[test]
    fn json_builder() {
        // Represents a student with some basic information
        #[derive(Debug)]
        #[allow(dead_code)]
        struct Student {
            name: String,
            age: u32,
            eye_color: EyeColor,
            grade_letter: String,
        }

        // The student's eye color
        #[derive(Debug)]
        enum EyeColor {
            Blue,
            Green,
            Brown,
        }

        // Implement the `FromJson` trait for `Student`, which provides the template for the expected JSON structure
        // and parsing logic for the resulting JSON.
        impl FromJson for Student {
            fn template() -> TemplateNode {
                object([
                    property("name", string()),
                    property("age", number(Some(0.0), None)),
                    property("eye_color", one_of(["blue", "green", "brown"])),
                    property("grade_letter", one_of(["<possible grades>"])),
                ])
            }

            fn from_json(json: &Map<String, JsonValue>) -> Result<Self> {
                let name = json["name"].as_str().unwrap().to_string();
                let age = json["age"].as_u64().unwrap() as u32;
                let eye_color = match json["eye_color"].as_str().unwrap() {
                    "blue" => EyeColor::Blue,
                    "green" => EyeColor::Green,
                    "brown" => EyeColor::Brown,
                    _ => unreachable!(), // The model should only be able to generate one of the three valid eye colors
                };
                let grade_letter = json["grade_letter"].as_str().unwrap().to_string();

                Ok(Student {
                    name,
                    age,
                    eye_color,
                    grade_letter,
                })
            }

            fn default_input_context() -> HashMap<String, String> {
                string_map! {
                    "possible grades" => "A|B|C|D|F",
                }
            }
        }

        const SEED: u64 = 13579;

        // Create the model
        let model = Model::new(ModelType::Qwen3Instruct(ModelSize::Small), SEED, true).unwrap();

        // Create a JSON builder
        let mut json_builder = JsonBuilder::new(model);

        // Build a few Student objects
        for _ in 0..5 {
            let student = json_builder
                .build::<Student>(
                    "Create a JSON object representing a fictional student.",
                    None,
                    Some(3),
                )
                .unwrap();

            println!("Generated student:\n{:#?}\n\n", student);
        }
    }

    #[test]
    fn chat_wrapper() {
        const SEED: u64 = 2125215;
        const TRIVIA_QUESTIONS: &[(&str, &str)] = &[
            ("What is the boiling point of water?", "humorous"),
            (
                "Who won the world series in 2020?",
                "very crass and aggressive",
            ),
        ];

        // Create the model
        let model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

        // Create the schemas for the chat wrapper
        // String slices also work as schemas
        let system_schema =
            "You are a quiz master who answers trivia questions in the provided tone.";
        let input_schema = ChatSchema::new()
            .with_text(Some("Question".to_string()), input_key("input"))
            .with_text(Some("Tone".to_string()), input_key("tone"));
        let output_schema = ChatSchema::new()
            .with_text(
                Some("Answer".to_string()),
                output_key("answer", Some("\""), &["\""]),
            )
            .with_text(
                Some("Explanation".to_string()),
                output_key("explanation", Some("\""), &["\""]),
            );

        // Create some examples
        let examples = [
            (
                string_map! {
                    "input" => "What is 2 + 2?",
                    "tone" => "pirate-like",
                },
                string_map! {
                    "answer" => "Arrr, 2 + 2 is 4.",
                    "explanation" => "This be because when ye add two and two together, ye get four. It be basic math, matey!",
                },
            ),
            (
                string_map! {
                    "input" => "What is the capital of Oregon?",
                    "tone" => "straightforward and non-verbose",
                },
                string_map! {
                    "answer" => "Salem.",
                    "explanation" => "Salem was made the capital of Oregon mainly due to its central location and accessibility.",
                },
            ),
        ];

        // Create the chat wrapper
        let mut chat_wrapper = SimpleChatWrapper::new(
            model,
            &InferParams::new_balanced(),
            system_schema,
            input_schema,
            output_schema,
            &examples,
            vec!["Answer concisely and accurately in the provided tone.".to_string()],
        )
        .0;

        // Get the output for each trivia question and print it
        for (question, tone) in TRIVIA_QUESTIONS {
            let input_context = string_map! {
                "input" => question,
                "tone" => tone,
            };
            let output = chat_wrapper.get_output(&input_context).into_captures();
            println!(
                "Question: {}\nTone: {}\nAnswer: {}\nExplanation: {}\n\n======\n",
                question, tone, output["answer"], output["explanation"]
            );
        }
    }

    #[test]
    fn chat() {
        const SEED: u64 = 477474;
        const CONVERSATION_TURNS: usize = 14;

        // Create the model
        let model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

        // Define tools
        let tools = vec![
            Tool::new(
                "list_files",
                "Lists the files in a directory.",
                [ParameterDefinition::new("directory", "The directory to list the files of.", ParameterType::String, None)],
                |args| {
                    let directory = args["directory"].as_str().unwrap();
                    let files = std::fs::read_dir(directory)
                        .unwrap()
                        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
                        .collect::<Vec<_>>()
                        .join(", ");
                    Ok(JsonValue::String(files))
                },
            ),
        ];

        // Start a chat
        let mut chat = Chat::new(
            model,
            "You are a helpful assistant and friendly person who has a great imagination, \
            an open mind, and is fun to talk to.",
            &[],
            &InferParams::new_balanced(),
            vec![
                "Be friendly and engaging in your responses.".to_string(),
                "Use your imagination to make the conversation more interesting.".to_string(),
            ],
            Some(string_map! {
                "your name" => "Quill",
                "the user's name" => "User",
            }),
            tools,
        )
        .0;

        // Infer a conversation
        for i in 0..CONVERSATION_TURNS {
            // Get the user message from the console input
            let mut input = String::new();
            print!("\nUser:\n");
            std::io::stdout().flush().unwrap();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim().to_string();

            // Push the user message to the chat history if it was not empty
            if !input.is_empty() {
                chat.push_message(ChatMessage::new(ChatRole::User, input));
            }

            // Infer a model response
            let message = chat.infer_message(&ChatRole::Assistant, None, &[]);
            let mut message_content = message.content().to_string();

            // If there was a malformed tool call then insert a warning before the message content
            if message.malformed_tool_call() {
                message_content = "[Warning: Malformed tool call]\n\n".to_string() + &message_content;
            }

            // If there was a tool call, execute the tool and append the assistant's response after the tool call to message_content
            if let Some(tool_call) = message.tool_call() {
                let (tool_response, response_message) = tool_call.execute(&mut chat).unwrap();
                println!("\n[Tool response:\n{}]", serde_json::to_string_pretty(&tool_response).unwrap());
                message_content.push_str(response_message.content());
            }

            // Print the assistant's response
            println!("\nAssistant:\n{}", message_content);

            // Compress the chat every other turn to test how stable it remains
            if i % 2 == 1 {
                chat.compress();
            }
        }
    }

    #[test]
    fn humanizer() {
        const SEED: u64 = 24680;
        const ITEMS_TO_JOIN: &[&[&str]] = &[
            &["the cat", "sat on", "the mat"],
            &["jack saw", "red roses", "beautiful sunset"],
            &["the quick", "brown fox", "jumps over", "the lazy dog."],
            &[
                "The ingredients for the recipe are",
                "flour",
                "sugar",
                "eggs",
                "milk",
                "butter",
                ".",
            ],
            &[
                "The main things I noticed about the movie are",
                "{\
                    \"acting\": \"great\",\n\
                    \"plot\": \"predictable\",\n\
                    \"cinematography\": \"stunning\",\n\
                    \"music\": \"forgettable\"\n\
                }",
            ],
            &[
                "The scene opens in",
                "a bustling city street filled with people and cars.",
                "Characters: [\
                    {\"name\": \"Alice\", \"role\": \"protagonist\", \"traits\": [\"brave\", \"curious\"]},\n\
                    {\"name\": \"Bob\", \"role\": \"antagonist\", \"traits\": [\"cunning\", \"ruthless\"]},\n\
                    {\"name\": \"Eve\", \"role\": \"sidekick\", \"traits\": [\"loyal\", \"resourceful\"]}\n\
                ]",
            ],
        ];

        // Create the model
        let model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

        // Create a humanizer
        let mut humanizer = Humanizer::new(model);

        for items in ITEMS_TO_JOIN {
            let result = humanizer.join(items);
            println!("Input: {}\nOutput: {}\n\n", items.join(" | "), result);
        }
    }

    #[test]
    fn generate_story() {
        const SEED: u64 = 547845;

        // Create the model
        let model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

        // Generate a story
        let mut story = "There was once".to_string();
        story.push_str(
            &model
                .predict_next(
                    "const long_story: string = \"There was once",
                    &InferParams::new_creative(),
                )
                .complete(&["\""])
                .unwrap(),
        );
        println!("Generated story:\n{}", story);
    }
}
