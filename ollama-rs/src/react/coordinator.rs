use std::collections::HashMap;

use crate::{
    error::OllamaError,
    generation::{
        chat::{request::ChatMessageRequest, ChatMessage, MessageRole},
        parameters::{FormatType, KeepAlive},
        tools::{Tool, ToolHolder, ToolInfo},
    },
    history::ChatHistory,
    models::ModelOptions,
    Ollama,
};

/// Default ReAct system prompt
const DEFAULT_SYSTEM_PROMPT: &str = r#"You are an AI assistant that uses the ReAct (Reasoning + Acting) framework to solve problems systematically.

For each user request, you should:
1. **Think** about what you need to do to solve the problem
2. **Act** by using available tools when you need external information or capabilities
3. **Observe** the results from your actions
4. Continue this cycle until you have a complete answer

When you respond, please structure your response as follows:
- Start with "Thought: " followed by your reasoning about the current situation
- Then use any tools you need to gather information
- I will show you the tool results, and you can continue thinking and acting until you're ready to provide a final answer

Be explicit about your thought process and use tools proactively when you need external information."#;

/// Represents a single step in the ReAct process
#[derive(Debug, Clone)]
pub struct Step {
    pub step_number: usize,
    pub thought: String,
    pub actions: Vec<String>,
    pub observations: Vec<String>,
    pub is_final: bool,
}

impl Step {
    pub fn new(step_number: usize) -> Self {
        Self {
            step_number,
            thought: String::new(),
            actions: Vec::new(),
            observations: Vec::new(),
            is_final: false,
        }
    }

    pub fn with_thought(mut self, thought: String) -> Self {
        self.thought = thought;
        self
    }

    pub fn add_action(mut self, action: String) -> Self {
        self.actions.push(action);
        self
    }

    pub fn add_observation(mut self, observation: String) -> Self {
        self.observations.push(observation);
        self
    }

    pub fn mark_final(mut self) -> Self {
        self.is_final = true;
        self
    }
}

/// A ReAct coordinator that works as an iterator, providing step-by-step
/// control over the reasoning and acting process.
pub struct Coordinator<C: ChatHistory + Default> {
    model: String,
    ollama: Ollama,
    options: ModelOptions,
    trajectory: C,
    tool_infos: Vec<ToolInfo>,
    tools: HashMap<String, Box<dyn ToolHolder>>,
    system_prompt: Option<String>,
    debug: bool,
    format: Option<FormatType>,
    keep_alive: Option<KeepAlive>,
    current_step: usize,
    max_iterations: usize,
    is_initialized: bool,
    think: Option<bool>,
}

impl<C: ChatHistory + Default> Coordinator<C> {
    /// Creates a new Coordinator with optional initial history
    pub fn new(ollama: Ollama, model: String, history: Option<C>) -> Self {
        let trajectory = history.unwrap_or_else(|| C::default());
        Self {
            model,
            ollama,
            options: ModelOptions::default(),
            trajectory,
            tool_infos: Vec::default(),
            tools: HashMap::default(),
            system_prompt: None,
            debug: false,
            format: None,
            keep_alive: None,
            current_step: 0,
            max_iterations: 10,
            is_initialized: false,
            think: None,
        }
    }

    /// Adds a tool to the coordinator
    pub fn add_tool<T: Tool + 'static>(mut self, tool: T) -> Self {
        self.tool_infos.push(ToolInfo::new::<_, T>());
        self.tools.insert(T::name().to_string(), Box::new(tool));
        self
    }

    /// Sets a custom system prompt (overrides the default ReAct prompt)
    pub fn system_prompt<S: Into<String>>(mut self, prompt: S) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the output format
    pub fn format(mut self, format: FormatType) -> Self {
        self.format = Some(format);
        self
    }

    /// Sets the model options
    pub fn options(mut self, options: ModelOptions) -> Self {
        self.options = options;
        self
    }

    /// Enables or disables debug mode
    pub fn debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    /// Sets the keep alive parameter
    pub fn keep_alive(mut self, keep_alive: KeepAlive) -> Self {
        self.keep_alive = Some(keep_alive);
        self
    }

    /// Sets the maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Sets thinking mode on/off
    pub fn think(mut self, think: bool) -> Self {
        self.think = Some(think);
        self
    }

    /// Starts the ReAct process with an initial user prompt
    pub async fn start(&mut self, prompt: &str) -> Result<(), crate::error::OllamaError> {
        if self.is_initialized {
            return Ok(());
        }

        let system_prompt = self
            .system_prompt
            .as_ref()
            .map(|s| s.as_str())
            .unwrap_or(DEFAULT_SYSTEM_PROMPT);

        self.trajectory
            .push(ChatMessage::system(system_prompt.to_string()));
        self.trajectory.push(ChatMessage::user(prompt.to_string()));
        self.is_initialized = true;

        if self.debug {
            eprintln!("Coordinator started with prompt: {}", prompt);
        }

        Ok(())
    }

    /// Generates the next ReAct step
    pub async fn next_step(&mut self) -> Result<Option<Step>, OllamaError> {
        if !self.is_initialized {
            return Err(OllamaError::Other(
                "Coordinator not initialized. Call start() first.".to_string(),
            ));
        }

        if self.current_step >= self.max_iterations {
            return Ok(None);
        }

        self.current_step += 1;

        if self.debug {
            eprintln!(
                "Coordinator step {}/{}",
                self.current_step, self.max_iterations
            );
        }

        let mut request = ChatMessageRequest::new(self.model.clone(), vec![])
            .options(self.options.clone())
            .tools(self.tool_infos.clone());

        if let Some(keep_alive) = &self.keep_alive {
            request = request.keep_alive(keep_alive.clone());
        }

        if let Some(think) = self.think {
            request = request.think(think);
        }

        // Only apply format if no tools or after tool execution
        if let Some(format) = &self.format {
            if self.tool_infos.is_empty() {
                request = request.format(format.clone());
            } else if let Some(last_message) = self.trajectory.messages().last() {
                if last_message.role == MessageRole::Tool {
                    request = request.format(format.clone());
                }
            }
        }

        let resp = self
            .ollama
            .send_chat_messages_with_history(&mut self.trajectory, request)
            .await?;

        if self.debug {
            eprintln!("Model response: {}", resp.message.content);
        }

        let mut react_step =
            Step::new(self.current_step).with_thought(resp.message.content.clone());

        // Check if there are tool calls to execute
        if !resp.message.tool_calls.is_empty() {
            for call in resp.message.tool_calls {
                if self.debug {
                    eprintln!("Executing tool: {:?}", call.function);
                }

                let action_description =
                    format!("{}({})", call.function.name, call.function.arguments);
                react_step = react_step.add_action(action_description);

                let Some(tool) = self.tools.get_mut(call.function.name.as_str()) else {
                    return Err(crate::error::ToolCallError::UnknownToolName.into());
                };

                let tool_result = tool
                    .call(call.function.arguments)
                    .await
                    .map_err(crate::error::ToolCallError::InternalToolError)?;

                if self.debug {
                    eprintln!("Tool result: {}", tool_result);
                }

                react_step = react_step.add_observation(tool_result.clone());
                self.trajectory.push(ChatMessage::tool(tool_result));
            }
        } else {
            // No tool calls - this might be the final answer
            react_step = react_step.mark_final();
        }

        Ok(Some(react_step))
    }

    /// Convenience method to collect all steps until completion
    pub async fn collect_all_steps(&mut self) -> Result<Vec<Step>, crate::error::OllamaError> {
        let mut steps = Vec::new();

        while let Some(step) = self.next_step().await? {
            let is_final = step.is_final;
            steps.push(step);
            if is_final {
                break;
            }
        }

        Ok(steps)
    }

    /// Gets the current conversation trajectory
    pub fn trajectory(&self) -> &C {
        &self.trajectory
    }

    /// Gets the current step number
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Checks if the coordinator has been initialized
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }
}
