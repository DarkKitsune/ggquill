use std::fmt::Display;

use crate::prelude::*;

/// Takes instructions/tasks in string form and breaks them down into numbered steps, potentially involving tools.
pub struct Director {
    chat_wrapper: SimpleChatWrapper,
}

impl Director {
    pub fn new(model: Model) -> Self {
        let system_schema = "You are an assistant. The user gives you instructions or complex tasks in string form and your job is to \
            breaks them down into a numbered list of steps.";

        // The input schema for the director is just a single instruction string
        let input_schema = ChatSchema::new()
            // First we start with the instruction string
            .with_text(Some("Instruction".to_string()), input_key("instruction"))
            // Then we include a list of tools available to the director, which is a stringified array of JSON tool definitions
            .with_text(Some("Available Tools".to_string()), input_key("tools"));

        // The output schema
        let output_schema = ChatSchema::new()
            // First we start the output with a list of steps needed to complete the task
            .with_text(
                Some("Steps".to_string()),
                output_key(
                    "steps",
                    Some("These are the steps needed to complete the task (in order):```\n"),
                    &["```"],
                ),
            );

        // The example tools used for generating example pairs
        let example_tools = [
            Tool::new(
                "get_user_email",
                "Retrieves the email address of the user.",
                [],
                |_args| Ok(()),
            ),
            Tool::new(
                "search",
                "Searches the internet based on a query string.",
                [ParameterDefinition::new(
                    "query",
                    "The search query.",
                    ParameterType::String,
                    None,
                )],
                |_args| Ok(()),
            ),
            Tool::new(
                "send_email",
                "Sends an email with the given recipient, subject and body at the specified time.",
                [
                    ParameterDefinition::new(
                        "recipient",
                        "The email address of the recipient.",
                        ParameterType::String,
                        None,
                    ),
                    ParameterDefinition::new(
                        "subject",
                        "The subject of the email.",
                        ParameterType::String,
                        None,
                    ),
                    ParameterDefinition::new(
                        "body",
                        "The body of the email.",
                        ParameterType::String,
                        None,
                    ),
                    ParameterDefinition::new(
                        "hour",
                        "The hour at which to send the email (0-23).",
                        ParameterType::Number,
                        Some(0.into()),
                    ),
                    ParameterDefinition::new(
                        "minute",
                        "The minute at which to send the email (0-59).",
                        ParameterType::Number,
                        Some(0.into()),
                    ),
                ],
                |_args| Ok(()),
            ),
        ];

        // The example pairs used to show the chat wrapper what a set of inputs and outputs should look like
        let tools_string = serde_json::to_string_pretty(
            &example_tools
                .iter()
                .map(Tool::to_json_schema)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let example_pairs = [
            (
                // Inputs
                string_map! {
                    "instruction" => "Find the current weather in New York and email it to me tomorrow morning.",
                    "tools" => tools_string,
                },
                // Outputs
                string_map! {
                    "steps" =>
                    "1. Find the current weather in New York.\n\
                    2. Get the user's email address.\n\
                    3. Email the weather information to the user tomorrow morning.",
                },
            ),
            (
                // Inputs
                string_map! {
                    "instruction" => "Search the internet for the best pizza places in Chicago, then search for the pizza place's menu, and email me the menu.",
                    "tools" => tools_string,
                },
                // Outputs
                string_map! {
                    "steps" =>
                    "1. Search the internet for the best pizza places in Chicago.\n\
                    2. Search the internet for the menu of the best pizza place in Chicago.\n\
                    3. Get the user's email address.\n\
                    4. Email the menu to the user.",
                },
            ),
            (
                // Inputs
                string_map! {
                    "instruction" => "Open a video about the history of the internet, and then search for the first word spoken in the video.",
                    "tools" => tools_string,
                },
                // Outputs
                string_map! {
                    "steps" => "This task cannot be completed with the available tools, as there is no tool for opening videos or extracting audio from videos.",
                },
            ),
        ];

        let chat_wrapper = SimpleChatWrapper::new(
                model,
                &InferParams::new_logical(),
                system_schema,
                input_schema,
                output_schema,
                &example_pairs,
                vec![
                    "Ensure that the steps are well-structured and clearly indicate the order of execution.".to_string(),
                    "Be creative where appropriate, yet accurate.".to_string(),
                    "Each step should be a single action that can be completed with one of the available tools, or by reasoning using the information provided.".to_string(),
                    "If the task cannot be completed with the available tools, provide an explanation of why it cannot be completed instead of steps.".to_string(),
                ]
            ).0;

        Self { chat_wrapper }
    }

    /// Takes in an instruction string and a list of available tools, and returns a vector list of steps needed to complete the task.
    pub fn get_steps(
        &mut self,
        instruction: impl Display,
        tools: &[Tool],
        opt_context: Option<&StringMap>,
    ) -> Result<Vec<String>> {
        // First make a vec of the tools' JSON schemas to present to the model
        let tools_schemas = tools.iter().map(Tool::to_json_schema).collect::<Vec<_>>();

        // Then we create the input for the chat wrapper by including the instruction and the tool schemas in the input map
        let mut input = string_map! {
            "instruction" => instruction.to_string().trim(),
            "tools" => serde_json::to_string_pretty(&tools_schemas).unwrap(),
        };

        // We also need to include the contents of opt_context in the input, so we extend the input map with the key-value pairs from opt_context
        if let Some(context) = opt_context {
            input.extend(context.clone());
        }

        // Finally we call the chat wrapper and get the output
        let output = self.chat_wrapper.get_output(&input).into_captures();

        // Extract the steps from the numbered list and remove the numbers
        let steps = output.get("steps").cloned().unwrap_or_default()
            .lines()
            .map(|line| line.trim().trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c.is_whitespace()).to_string())
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>();

        Ok(steps)
    }
}
