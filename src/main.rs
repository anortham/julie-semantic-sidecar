#[cfg(any(unix, windows))]
use julie_semantic_sidecar::broker::{self, BrokerConfig, BrokerEndpoint};
use julie_semantic_sidecar::{prepare, protocol, DEFAULT_MODEL_ID, VERSION};
#[cfg(any(unix, windows))]
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
usage: julie-semantic-sidecar [serve [--model <id>] | prepare [--model <id>] | broker --model <id> --endpoint <path> --lock <path> --accelerator-lock <path> | --version]

  serve [--model <id>]     speak the julie.embedding.sidecar v1 protocol on stdio (default verb)
  prepare [--model <id>]   download and verify a manifest model into the shared cache
  broker ...               share one model over a current-user local endpoint
  --version                print the binary version
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match Cli::parse(&args) {
        Ok(Cli::Version) => {
            println!("julie-semantic-sidecar {VERSION}");
            ExitCode::SUCCESS
        }
        Ok(Cli::Serve { model }) => match protocol::serve(&model) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("julie-semantic-sidecar: serve failed: {err}");
                ExitCode::FAILURE
            }
        },
        Ok(Cli::Prepare { model }) => prepare::run(model.as_deref()),
        Ok(Cli::Broker {
            model,
            endpoint,
            service_lock,
            accelerator_lock,
        }) => run_broker(model, endpoint, service_lock, accelerator_lock),
        Err(err) => {
            eprintln!("julie-semantic-sidecar: {err}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(unix)]
fn run_broker(
    model: String,
    endpoint: String,
    service_lock: String,
    accelerator_lock: String,
) -> ExitCode {
    let config = BrokerConfig {
        model_id: model,
        endpoint: BrokerEndpoint::Unix(PathBuf::from(endpoint)),
        service_lock: PathBuf::from(service_lock),
        accelerator_lock: PathBuf::from(accelerator_lock),
    };
    match broker::serve(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("julie-semantic-sidecar: broker failed: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn run_broker(
    model: String,
    endpoint: String,
    service_lock: String,
    accelerator_lock: String,
) -> ExitCode {
    let config = BrokerConfig {
        model_id: model,
        endpoint: BrokerEndpoint::Windows(endpoint),
        service_lock: PathBuf::from(service_lock),
        accelerator_lock: PathBuf::from(accelerator_lock),
    };
    match broker::serve(config) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("julie-semantic-sidecar: broker failed: {err}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(not(any(unix, windows)))]
fn run_broker(
    _model: String,
    _endpoint: String,
    _service_lock: String,
    _accelerator_lock: String,
) -> ExitCode {
    eprintln!("julie-semantic-sidecar: broker transport is not available on this platform");
    ExitCode::FAILURE
}

#[derive(Debug, PartialEq, Eq)]
enum Cli {
    Serve {
        model: String,
    },
    Prepare {
        model: Option<String>,
    },
    Broker {
        model: String,
        endpoint: String,
        service_lock: String,
        accelerator_lock: String,
    },
    Version,
}

impl Cli {
    fn parse(args: &[String]) -> Result<Self, String> {
        let (verb, rest) = match args.split_first() {
            None => {
                return Ok(Cli::Serve {
                    model: DEFAULT_MODEL_ID.to_string(),
                })
            }
            Some((verb, rest)) => (verb.as_str(), rest),
        };
        match verb {
            "--version" | "-V" | "version" => {
                if rest.is_empty() {
                    Ok(Cli::Version)
                } else {
                    Err(format!("unexpected argument: {}", rest[0]))
                }
            }
            "serve" => Ok(Cli::Serve {
                model: parse_model(rest)?.unwrap_or_else(|| DEFAULT_MODEL_ID.to_string()),
            }),
            "prepare" => Ok(Cli::Prepare {
                model: parse_model(rest)?,
            }),
            "broker" => parse_broker(rest),
            other => Err(format!("unknown verb: {other}")),
        }
    }
}

fn parse_broker(args: &[String]) -> Result<Cli, String> {
    let mut model = None;
    let mut endpoint = None;
    let mut service_lock = None;
    let mut accelerator_lock = None;
    let mut index = 0;
    while index < args.len() {
        let flag = &args[index];
        let value = args
            .get(index + 1)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| format!("{flag} requires a value"))?;
        let target = match flag.as_str() {
            "--model" => &mut model,
            "--endpoint" => &mut endpoint,
            "--lock" => &mut service_lock,
            "--accelerator-lock" => &mut accelerator_lock,
            _ => return Err(format!("unexpected argument: {flag}")),
        };
        if target.replace(value.clone()).is_some() {
            return Err(format!("duplicate argument: {flag}"));
        }
        index += 2;
    }
    Ok(Cli::Broker {
        model: model.ok_or_else(|| "broker requires --model".to_string())?,
        endpoint: endpoint.ok_or_else(|| "broker requires --endpoint".to_string())?,
        service_lock: service_lock.ok_or_else(|| "broker requires --lock".to_string())?,
        accelerator_lock: accelerator_lock
            .ok_or_else(|| "broker requires --accelerator-lock".to_string())?,
    })
}

fn parse_model(args: &[String]) -> Result<Option<String>, String> {
    match args {
        [] => Ok(None),
        [flag, value] if flag == "--model" => {
            if value.is_empty() {
                Err("--model requires a manifest model id".to_string())
            } else {
                Ok(Some(value.clone()))
            }
        }
        [flag] if flag == "--model" => Err("--model requires a manifest model id".to_string()),
        [first, ..] => Err(format!("unexpected argument: {first}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Cli, String> {
        let owned: Vec<String> = args.iter().map(|a| (*a).to_string()).collect();
        Cli::parse(&owned)
    }

    #[test]
    fn no_args_serves_the_default_model() {
        assert_eq!(
            parse(&[]),
            Ok(Cli::Serve {
                model: "bge-small-en-v1.5-f32".to_string()
            })
        );
    }

    #[test]
    fn serve_without_a_model_uses_bge() {
        assert_eq!(
            parse(&["serve"]),
            Ok(Cli::Serve {
                model: "bge-small-en-v1.5-f32".to_string()
            })
        );
    }

    #[test]
    fn serve_accepts_the_explicit_qwen_comparison_model() {
        assert_eq!(
            parse(&["serve", "--model", "qwen3-0.6b-f16"]),
            Ok(Cli::Serve {
                model: "qwen3-0.6b-f16".to_string()
            })
        );
    }

    #[test]
    fn prepare_defaults_to_no_explicit_model() {
        assert_eq!(parse(&["prepare"]), Ok(Cli::Prepare { model: None }));
    }

    #[test]
    fn prepare_accepts_a_model_override() {
        assert_eq!(
            parse(&["prepare", "--model", "qwen3-0.6b-f16"]),
            Ok(Cli::Prepare {
                model: Some("qwen3-0.6b-f16".to_string())
            })
        );
    }

    #[test]
    fn broker_requires_and_parses_all_four_contract_arguments() {
        assert_eq!(
            parse(&[
                "broker",
                "--model",
                "bge-small-en-v1.5-f32",
                "--endpoint",
                "/tmp/broker.sock",
                "--lock",
                "/tmp/broker.lock",
                "--accelerator-lock",
                "/tmp/accelerator.lock",
            ]),
            Ok(Cli::Broker {
                model: "bge-small-en-v1.5-f32".to_string(),
                endpoint: "/tmp/broker.sock".to_string(),
                service_lock: "/tmp/broker.lock".to_string(),
                accelerator_lock: "/tmp/accelerator.lock".to_string(),
            })
        );
    }

    #[test]
    fn broker_rejects_a_missing_contract_argument() {
        assert!(parse(&[
            "broker",
            "--model",
            "bge-small-en-v1.5-f32",
            "--endpoint",
            "/tmp/broker.sock",
            "--lock",
            "/tmp/broker.lock",
        ])
        .is_err());
    }

    #[test]
    fn version_flag_parses() {
        assert_eq!(parse(&["--version"]), Ok(Cli::Version));
    }

    #[test]
    fn unknown_verb_is_rejected() {
        assert!(parse(&["embed"]).is_err());
    }

    #[test]
    fn model_flag_without_value_is_rejected() {
        assert!(parse(&["serve", "--model"]).is_err());
    }
}
