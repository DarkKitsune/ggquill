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
    use std::io::Write;

    use crate::{
        chat_wrapper::{ChatWrapper, SimpleChatWrapper},
        prelude::*,
    };

    #[test]
    fn json_builder() {
        const SEED: u64 = 13579;

        // Create the model
        let mut model = Model::new(ModelType::Qwen3Instruct(ModelSize::Small), SEED, true).unwrap();

        // Create a JSON builder
        let mut json_builder = JsonBuilder::new(&mut model);

        // Define a template for the expected JSON structure
        
        let template = object([
            property("scene", object([
                property("player", object([
                    property("health", number(Some(0.0), Some(100.0))),
                    property("position", object([
                        property("x", number(None, None)),
                        property("y", number(None, None)),
                        property("z", number(None, None)),
                    ])),
                ])),
                property("enemies", array(object([
                    property("name", string()),
                    property("health", number(Some(0.0), Some(100.0))),
                    property("position", object([
                        property("x", number(None, None)),
                        property("y", number(None, None)),
                        property("z", number(None, None)),
                    ])),
                ]))),
            ])),
        ]);

        println!("Template for expected JSON structure:\n{}\n", template);

        // Define some instructions for building JSON and print the generated JSON outputs
        let instructions_list = [
            "Build a JSON object representing a scene tree for an interesting horror game with 'enemies' populated by lovecraftian monsters, \
            The player should be in good health and be centered at the origin.",
        ];

        for instructions in instructions_list {
            let output_json = json_builder.build_json(instructions, &template, Some(5)).unwrap();
            println!(
                "\nInstructions: {}\nGenerated JSON:\n{}\n",
                instructions,
                serde_json::to_string_pretty(&output_json).unwrap()
            );
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
        let mut model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

        // Create the schemas for the chat wrapper
        // String slices also work as schemas
        let system_schema =
            "You are a quiz master who answers trivia questions in the provided tone.";
        let input_schema = ChatSchema::new()
            .with_text(Some("Question".to_string()), "<input>")
            .with_text(Some("Tone".to_string()), "Please answer in a <tone> tone.");
        let output_schema = ChatSchema::new()
            .with_text(Some("Answer".to_string()), "\"<\" :: answer>")
            .with_text(Some("Explanation".to_string()), "\"<\" :: explanation>");

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
            &mut model,
            &InferParams::new_balanced(),
            system_schema,
            input_schema,
            output_schema,
            &examples,
            vec!["Answer concisely and accurately, and explain your answer further.".to_string()],
        );

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

        // Print timings
        println!(
            "\n\nAverage tok/s: {}",
            model.average_tokens_per_second().unwrap()
        );
    }

    #[test]
    fn chat() {
        const SEED: u64 = 477474;
        const CONVERSATION_TURNS: usize = 14;

        // Create the model
        let mut model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

        // Start a chat
        let mut chat = Chat::new(
            &mut model,
            "You are a helpful assistant and friendly person who has a great imagination, \
            an open mind, and is fun to talk to.",
            &[],
            &InferParams::new_creative(),
            vec![
                "Be friendly and engaging in your responses.".to_string(),
                "Use your imagination to make the conversation more interesting.".to_string(),
            ],
            Some(string_map! {
                "your name" => "Quill",
                "the user's name" => "User",
            }),
        );

        // Infer a conversation
        for _ in 0..CONVERSATION_TURNS {
            // Get the user message from the console input
            let mut input = String::new();
            print!("User: ");
            std::io::stdout().flush().unwrap();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim().to_string();

            // Push the user message to the chat history
            chat.push_message(ChatMessage::new(ChatRole::User, input));

            // Infer a model response
            let message = chat.infer_message(&ChatRole::Assistant, None, &[]);

            println!("\n\n\nAssistant: {}\n\n\n", message);
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
        let mut model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

        // Create a humanizer
        let mut humanizer = Humanizer::new(&mut model);

        for items in ITEMS_TO_JOIN {
            let result = humanizer.join(items);
            println!("Input: {}\nOutput: {}\n\n", items.join(" | "), result);
        }
    }

    #[test]
    fn generate_story() {
        const SEED: u64 = 547845;

        // Create the model
        let mut model = Model::new(ModelType::Qwen3Instruct(ModelSize::Medium), SEED, true).unwrap();

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
        println!(
            "\n\nAverage tok/s: {}",
            model.average_tokens_per_second().unwrap()
        );
    }
}
