pub mod actor;
pub mod chat;
pub mod chat_schema;
pub mod chat_wrapper;
pub mod data;
pub mod humanizer;
pub mod inference;
pub mod instructor;
pub mod model;
pub mod model_type;
pub mod pipeline;
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
        let mut model = Model::new(ModelType::Qwen3InstructQuantized, SEED, true).unwrap();

        // Create the schemas for the chat wrapper
        // String slices also work as schemas
        let system_schema = "You are a quiz master who answers trivia questions in the provided tone. \
            Answer concisely and accurately, and explain your answer further.";
        let input_schema = ChatSchema::new()
            .with_text(Some("Question".to_string()), "{input}")
            .with_text(Some("Tone".to_string()), "Please answer in a {tone} tone.");
        let output_schema = ChatSchema::new()
            .with_text(Some("Answer".to_string()), "\"{\" => answer}")
            .with_text(Some("Explanation".to_string()), "\"{\" => explanation}");

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
            examples,
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
    fn pipeline() {
        const SEED: u64 = 46364;
        const POEM_THEMES: &[&str] = &[
            "a magical adventure in a fantasy world",
            "the beauty of nature in spring",
            "breaking free from constraints and embracing freedom",
        ];

        // Create the model and pipeline
        let mut model = Model::new(ModelType::Qwen3InstructQuantized, SEED, true).unwrap();
        let mut pipeline = Pipeline::new(&mut model);

        // Add a system propmt instructing the model to act as a poet
        pipeline.system_prompt(
            "You are a talented poet who writes beautiful and creative poems. \
            Your poems should be 2 or 3 stanzas long and rhyme.",
        );

        // Add an instruct step to the pipeline which generates a poem based on the theme in {theme}
        // And store the generated poem in the context under {poem}
        pipeline.instruct(
            "poem",
            "Write a short poem based on the following theme: {theme}",
            Some("Here is the poem:\n".to_string()),
            vec![],
        );

        // Add a system prompt instructing the model to summarize poems in a concise manner while preserving the core meaning and style of the original poem
        pipeline.system_prompt(
            "When summarizing a poem, you should create a concise version of the poem that preserves the core \
                meaning and style of the original poem. The summary should be much shorter than the original poem."
        );

        // Add a step to the pipeline which summarizes the poem in {poem} and stores the summary in the context under {summary}
        pipeline.summarize("summary", "{poem}", "Concise and easy to understand");

        // Execute the pipeline on the model for each poem theme and print the poem and summary outputs
        for theme in POEM_THEMES {
            let mut context = json_map! {
                "theme" => *theme,
            };
            pipeline.execute(&mut context);
            let poem = context["poem"].as_str().unwrap();
            let summary = context["summary"].as_str().unwrap();
            println!(
                "Theme: {}\n\nGenerated poem:\n{}\n\n\n\nSummarized version:\n{}\n\n\n\n------\n\n\n",
                theme, poem, summary
            );
        }
    }

    #[test]
    fn chat() {
        const SEED: u64 = 477474;
        const CONVERSATION_TURNS: usize = 14;

        // Create the model
        let mut model = Model::new(ModelType::Qwen3InstructQuantized, SEED, true).unwrap();

        // Start a chat
        let mut chat = Chat::new(
            &mut model,
            "You are a helpful assistant and friendly person who has a great imagination, \
            an open mind, and is fun to talk to. Talk to the user in a friendly and engaging manner.",
            &[],
            &InferParams::new_creative(),
            None,
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
        let mut model = Model::new(ModelType::Qwen3InstructQuantized, SEED, true).unwrap();

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
        let mut model = Model::new(ModelType::Qwen3InstructQuantized, SEED, true).unwrap();

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
