pub mod actor;
pub mod chat;
pub mod data;
pub mod inference;
pub mod instructor;
pub mod joiner;
pub mod model;
pub mod model_type;
pub mod pipeline;
pub mod prelude;
pub mod token_string;

#[cfg(test)]
mod tests {
    use std::io::Write;

    use crate::prelude::*;

    #[test]
    fn pipeline() {
        const SEED: u64 = 46364;
        const POEM_THEMES: &[&str] = &[
            "a magical adventure in a fantasy world",
            "the beauty of nature in spring",
            "breaking free from constraints and embracing freedom",
        ];

        // Create the model and pipeline
        let mut model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();
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
                "theme" => theme,
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
        let mut model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

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
    fn predict_chain() {
        const SEED: u64 = 13579;

        // Create the model
        let mut model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        let mut prediction = model.predict_next(
            "Here is my character bio:\nName: Jessie\nAge: 19\nClass: Archer",
            &InferParams::new_balanced(),
        );
        prediction.push_str("\nWeapon: ");
        let weapon = prediction.next_value();
        prediction.push_str("\nClothing: ");
        let clothing = prediction.next_value();
        prediction.push_str("\nHometown: ");
        let hometown = prediction.next_value();
        println!(
            "Predicted character bio:\nName: Jessie\nAge: 19\nClass: Archer\nWeapon: {}\nClothing: {}\nHometown: {}",
            weapon, clothing, hometown
        );
    }

    #[test]
    fn joiner() {
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
                "To get to",
                "a park",
                "go",
                "straight",
                "two blocks",
                "turn left",
                "on right",
                ".",
            ],
            &[
                "The weather today is",
                "sunny",
                "with a high of 75 degrees",
                "and a low of 55 degrees",
                ".",
            ],
            &[
                "func",
                "test(",
                "foo: number",
                ") {",
                "return foo * 2;",
                "}",
            ],
        ];

        // Create the model
        let mut model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Create a joiner
        let mut joiner = Joiner::new(&mut model);

        for items in ITEMS_TO_JOIN {
            let result = joiner.join(items);
            println!("{:?}\n{}\n\n", items, result);
        }
    }

    #[test]
    fn generate_story() {
        const SEED: u64 = 547845;

        // Create the model
        let mut model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Generate a story
        let mut story = "There was once".to_string();
        story.push_str(
            &model
                .predict_next("story = \"There was once", &InferParams::new_creative())
                .complete(&["\""])
                .unwrap(),
        );
        println!("Generated story:\n{}", story);
    }

    #[test]
    fn instructor() {
        const SEED: u64 = 98765;
        const INSTRUCTIONS: &[&str] = &[
            "Email Jessie a snarky message about how she is the worst.",
            "Walk my dog Max in the park at 5 PM tomorrow.",
            "Order a pizza from Cosmic Joe's with pepperoni, mushrooms, and extra cheese.",
            "Remind me to call Alice about the project update at 3 PM today. Subject: Project Update Reminder.",
        ];

        // Create the model
        let mut model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Define some instruction definitions for the instructor to learn from
        let instruction_definitions = vec![
            InstructionDefinition::new(
                "send_email",
                vec!["recipient".into(), "subject".into(), "body".into()],
                "Send a short email to John reminding him about the meeting tomorrow at half past 10.",
                vec![
                    "John".into(),
                    "Meeting Reminder".into(),
                    "Don't forget about our meeting tomorrow at 10:30 AM!".into(),
                ],
            ),
            InstructionDefinition::new(
                "travel",
                vec!["destination".into(), "when".into(), "travel_method".into()],
                "I'd like to drive my red car to Paris tomorrow.",
                vec!["Paris".into(), "tomorrow".into(), "car".into()],
            ),
            InstructionDefinition::new(
                "order_food",
                vec!["from_where".into(), "dish".into()],
                "Yesterday I went to the burger place down the street. I got the bbq bacon burger and it was delicious!",
                vec![
                    "the burger place down the street".into(),
                    "bbq bacon burger".into(),
                ],
            ),
        ];

        // Create an instructor with the instruction definitions
        let mut instructor = Instructor::new(&mut model, instruction_definitions);

        // Parse each instruction in INSTRUCTIONS and print the resulting function calls
        for instruction in INSTRUCTIONS {
            let parsed_instruction = instructor.parse_instruction(instruction).unwrap();
            println!(
                "Instruction:\n{}\n\nParsed instruction:\nName: {}\nArgs: {:?}\n\n\n",
                instruction, parsed_instruction.name, parsed_instruction.args
            );
        }
    }
}
