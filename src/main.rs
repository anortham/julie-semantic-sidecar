use julie_semantic_sidecar::{prepare, protocol, DEFAULT_MODEL_ID, VERSION};
use std::process::ExitCode;

const USAGE: &str = "\
usage: julie-semantic-sidecar [serve [--model <id>] | prepare [--model <id>] | --version]

  serve [--model <id>]     speak the julie.embedding.sidecar v1 protocol on stdio (default verb)
  prepare [--model <id>]   download and verify a manifest model into the shared cache
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
        Err(err) => {
            eprintln!("julie-semantic-sidecar: {err}");
            eprint!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Cli {
    Serve { model: String },
    Prepare { model: Option<String> },
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
            other => Err(format!("unknown verb: {other}")),
        }
    }
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
                model: DEFAULT_MODEL_ID.to_string()
            })
        );
    }

    #[test]
    fn serve_accepts_a_model_override() {
        assert_eq!(
            parse(&["serve", "--model", "bge-small-en-v1.5-f32"]),
            Ok(Cli::Serve {
                model: "bge-small-en-v1.5-f32".to_string()
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
