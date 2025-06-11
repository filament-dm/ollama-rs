use ollama_rs::{react::Coordinator, Ollama};

const TEST_MODEL: &str = "qwen3:0.6b";

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct GetTemperatureParams {
    unit: Option<String>,
}

struct GetTemperature;

impl ollama_rs::generation::tools::Tool for GetTemperature {
    type Params = GetTemperatureParams;

    fn name() -> &'static str {
        "get_temperature"
    }

    fn description() -> &'static str {
        "Get the current temperature in Fahrenheit or Celsius."
    }

    async fn call(
        &mut self,
        parameters: Self::Params,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let unit = parameters.unit.unwrap_or_else(|| "fahrenheit".to_string());
        let unit_lower = unit.to_lowercase();
        if unit_lower == "celsius" || unit_lower == "c" {
            Ok("21.6".to_string())
        } else {
            Ok("71".to_string())
        }
    }
}

#[derive(serde::Deserialize, schemars::JsonSchema)]
struct CalculateParams {
    operation: String,
    a: f64,
    b: f64,
}

struct Calculator;

impl ollama_rs::generation::tools::Tool for Calculator {
    type Params = CalculateParams;

    fn name() -> &'static str {
        "calculate"
    }

    fn description() -> &'static str {
        "Perform basic math operations (add, subtract, multiply, divide) on two numbers."
    }

    async fn call(
        &mut self,
        parameters: Self::Params,
    ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let result = match parameters.operation.as_str() {
            "add" | "+" => parameters.a + parameters.b,
            "subtract" | "-" => parameters.a - parameters.b,
            "multiply" | "*" => parameters.a * parameters.b,
            "divide" | "/" => {
                if parameters.b != 0.0 {
                    parameters.a / parameters.b
                } else {
                    return Ok("Error: Division by zero".to_string());
                }
            }
            _ => return Ok("Error: Unknown operation".to_string()),
        };
        Ok(result.to_string())
    }
}

#[tokio::test]
async fn test_react_coordinator_basic_iteration() {
    let ollama = Ollama::default();
    let mut coordinator = Coordinator::<Vec<_>>::new(ollama, TEST_MODEL.to_string(), None)
        .debug(true)
        .add_tool(GetTemperature)
        .max_iterations(5);

    assert!(
        coordinator
            .start("What's the current temperature?")
            .await
            .is_ok(),
        "Start should not fail"
    );

    assert!(coordinator.is_initialized());
    assert_eq!(coordinator.current_step(), 0);

    let step_result = coordinator.next_step().await;
    match step_result {
        Ok(Some(step)) => {
            println!("Step {}: {}", step.step_number, step.thought);

            // this should be a single step call
            assert_eq!(step.step_number, 1);

            // tools should have been called
            assert!(!step.thought.is_empty());
            if !step.actions.is_empty() {
                assert!(!step.observations.is_empty());
                println!("Actions: {:?}", step.actions);
                println!("Observations: {:?}", step.observations);

                // should contain temperature information
                let observations_text = step.observations.join(" ");
                assert!(observations_text.contains("71") || observations_text.contains("21.6"));
            }
        }
        Ok(None) => {
            println!("No steps returned - max iterations reached");
            assert!(false);
        }
        Err(e) => {
            assert!(false, "Ollama call failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_react_coordinator_multiple_steps() {
    let ollama = Ollama::default();
    let mut coordinator = Coordinator::<Vec<_>>::new(ollama, TEST_MODEL.to_string(), None)
        .debug(true)
        .add_tool(GetTemperature)
        .add_tool(Calculator)
        .max_iterations(3);

    assert!(
        coordinator
            .start("What's the current temperature and what would it be if we added 10 degrees?")
            .await
            .is_ok(),
        "Start should not fail"
    );

    let mut step_count = 0;
    loop {
        match coordinator.next_step().await {
            Ok(Some(step)) => {
                step_count += 1;
                println!("\nStep {}: {}", step.step_number, step.thought);

                if !step.actions.is_empty() {
                    println!("Actions: {:?}", step.actions);
                    println!("Observations: {:?}", step.observations);
                }

                if step.is_final {
                    println!("Final step reached");
                    break;
                }
            }
            Ok(None) => {
                println!("Max iterations reached");
                break;
            }
            Err(e) => {
                assert!(false, "Step iteration should not fail: {:?}", e);
            }
        }
    }

    println!("Completed {} steps", step_count);
    assert!(step_count >= 1);
}

#[tokio::test]
async fn test_react_coordinator_collect_all_steps() {
    let ollama = Ollama::default();
    let mut coordinator = Coordinator::<Vec<_>>::new(ollama, TEST_MODEL.to_string(), None)
        .debug(true)
        .add_tool(Calculator)
        .max_iterations(5);

    assert!(
        coordinator.start("Calculate 15 times 3").await.is_ok(),
        "Start should not fail"
    );

    let steps_result = coordinator.collect_all_steps().await;
    match steps_result {
        Ok(steps) => {
            println!("Collected {} steps", steps.len());

            for (i, step) in steps.iter().enumerate() {
                println!("\nStep {}: {}", i + 1, step.thought);
                if !step.actions.is_empty() {
                    println!("  Actions: {:?}", step.actions);
                    println!("  Observations: {:?}", step.observations);
                }
            }

            assert!(!steps.is_empty());

            if let Some(last_step) = steps.last() {
                assert!(last_step.is_final);
            }

            let used_calculator = steps.iter().any(|step| {
                step.actions
                    .iter()
                    .any(|action| action.contains("calculate"))
            });

            if used_calculator {
                // Should have the result 45
                let has_result = steps
                    .iter()
                    .any(|step| step.observations.iter().any(|obs| obs.contains("45")));
                assert!(has_result, "Should contain calculation result");
            }
        }
        Err(e) => {
            assert!(false, "collect_all_steps should not fail: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_react_coordinator_max_iterations() {
    let ollama = Ollama::default();
    let mut coordinator = Coordinator::<Vec<_>>::new(ollama, TEST_MODEL.to_string(), None)
        .debug(true)
        .max_iterations(2);

    assert!(
        coordinator.start("Count from 1 to 100").await.is_ok(),
        "Start should not fail"
    );

    let mut step_count = 0;
    loop {
        match coordinator.next_step().await {
            Ok(Some(step)) => {
                step_count += 1;
                println!("Step {}: {}", step.step_number, step.thought);

                if step.is_final {
                    break;
                }
            }
            Ok(None) => {
                println!("Reached max iterations as expected");
                break;
            }
            Err(e) => {
                assert!(false, "Step iteration should not fail: {:?}", e);
            }
        }
    }

    assert!(step_count <= 2, "Should not exceed max iterations");
    println!("Completed {} steps (max: 2)", step_count);
}

#[tokio::test]
async fn test_react_coordinator_without_tools() {
    let ollama = Ollama::default();
    let mut coordinator = Coordinator::<Vec<_>>::new(ollama, TEST_MODEL.to_string(), None)
        .debug(true)
        .max_iterations(3);

    assert!(
        coordinator.start("What is 2 + 2?").await.is_ok(),
        "Start should not fail"
    );

    match coordinator.next_step().await {
        Ok(Some(step)) => {
            println!("Step {}: {}", step.step_number, step.thought);

            // if no tools we should be marking as final immediately
            assert!(step.is_final);
            assert!(step.actions.is_empty());
            assert!(step.observations.is_empty());
            assert!(!step.thought.is_empty());
        }
        Ok(None) => {
            println!("No steps returned");
        }
        Err(e) => {
            assert!(false, "Step should not fail: {:?}", e);
        }
    }
}
