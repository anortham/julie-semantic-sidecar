use julie_semantic_sidecar::broker::engine::BrokerEngine;
use julie_semantic_sidecar::broker::lease::AcceleratorLease;
use julie_semantic_sidecar::engine::BackendPolicy;
use julie_semantic_sidecar::engine_trait::{
    EmbedEngine, EmbedOutput, EngineError, EngineFailureClass, Role,
};
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::path::Path;
use std::rc::Rc;

const RUNTIME_DEGRADATION: &str = "accelerator resource exhausted; permanently demoted to CPU";

#[derive(Clone, Copy)]
enum Outcome {
    Success,
    ResourceExhausted,
    Application(&'static str),
}

struct FakeEngine {
    backend: &'static str,
    accelerated: bool,
    outcomes: RefCell<VecDeque<Outcome>>,
    calls: Rc<Cell<usize>>,
    drops: Option<Rc<RefCell<Vec<&'static str>>>>,
}

impl FakeEngine {
    fn new(
        backend: &'static str,
        accelerated: bool,
        outcomes: impl IntoIterator<Item = Outcome>,
        calls: Rc<Cell<usize>>,
    ) -> Self {
        Self {
            backend,
            accelerated,
            outcomes: RefCell::new(outcomes.into_iter().collect()),
            calls,
            drops: None,
        }
    }

    fn with_drop_log(mut self, drops: Rc<RefCell<Vec<&'static str>>>) -> Self {
        self.drops = Some(drops);
        self
    }
}

impl Drop for FakeEngine {
    fn drop(&mut self) {
        if let Some(drops) = &self.drops {
            drops.borrow_mut().push(self.backend);
        }
    }
}

impl EmbedEngine for FakeEngine {
    fn health_facts(&self) -> Result<Value, EngineError> {
        Ok(json!({
            "ready": true,
            "resolved_backend": self.backend,
            "accelerated": self.accelerated,
            "degraded_reason": null,
            "load_policy": {
                "requested_device_backend": self.backend,
                "resolved_device_backend": self.backend,
                "accelerated": self.accelerated,
                "degraded_reason": null
            }
        }))
    }

    fn is_accelerated(&self) -> bool {
        self.accelerated
    }

    fn embed(&self, texts: &[String], _role: Role) -> Result<EmbedOutput, EngineError> {
        self.calls.set(self.calls.get() + 1);
        match self
            .outcomes
            .borrow_mut()
            .pop_front()
            .unwrap_or(Outcome::Success)
        {
            Outcome::Success => Ok(EmbedOutput {
                dims: 1,
                vectors: texts.iter().map(|_| vec![1.0]).collect(),
            }),
            Outcome::ResourceExhausted => Err(EngineError::resource_exhausted(
                "ContextAlloc",
                "out of memory",
            )),
            Outcome::Application(kind) => Err(EngineError::new(kind, "out of memory")),
        }
    }
}

fn acquire(path: &Path) -> AcceleratorLease {
    AcceleratorLease::try_acquire(path)
        .expect("accelerator lock is usable")
        .expect("accelerator lock is free")
}

#[test]
fn broker_accelerator_resource_exhaustion_reloads_cpu_once_releases_lease_and_stays_cpu() {
    let temp = tempfile::tempdir().expect("tempdir");
    let lock_path = temp.path().join("accelerator.lock");
    let policies = Rc::new(RefCell::new(Vec::new()));
    let accelerated_calls = Rc::new(Cell::new(0));
    let cpu_calls = Rc::new(Cell::new(0));
    let drops = Rc::new(RefCell::new(Vec::new()));

    let engine = BrokerEngine::load_with(acquire(&lock_path).into(), {
        let policies = Rc::clone(&policies);
        let accelerated_calls = Rc::clone(&accelerated_calls);
        let cpu_calls = Rc::clone(&cpu_calls);
        let drops = Rc::clone(&drops);
        let lock_path = lock_path.clone();
        move |policy| {
            policies.borrow_mut().push(policy);
            Ok(match policy {
                BackendPolicy::Auto => FakeEngine::new(
                    "metal",
                    true,
                    [Outcome::ResourceExhausted],
                    Rc::clone(&accelerated_calls),
                )
                .with_drop_log(Rc::clone(&drops)),
                BackendPolicy::CpuOnly => {
                    assert_eq!(drops.borrow().as_slice(), ["metal"]);
                    drop(acquire(&lock_path));
                    FakeEngine::new("cpu", false, [Outcome::Success], Rc::clone(&cpu_calls))
                }
            })
        }
    })
    .expect("accelerated engine loads");

    let first = engine
        .embed(&["query".to_string()], Role::Query)
        .expect("request retries on CPU");
    assert_eq!(first.vectors, vec![vec![1.0]]);
    assert_eq!(accelerated_calls.get(), 1);
    assert_eq!(cpu_calls.get(), 1);
    assert_eq!(
        policies.borrow().as_slice(),
        [BackendPolicy::Auto, BackendPolicy::CpuOnly]
    );
    assert!(!engine.accelerator_lease_held());
    drop(acquire(&lock_path));

    engine
        .embed(&["later".to_string()], Role::Query)
        .expect("later request stays on CPU");
    assert_eq!(accelerated_calls.get(), 1);
    assert_eq!(cpu_calls.get(), 2);
    assert_eq!(policies.borrow().len(), 2);
}

#[test]
fn broker_accelerator_two_model_identities_cannot_both_construct_accelerated_engines() {
    let temp = tempfile::tempdir().expect("tempdir");
    let lock_path = temp.path().join("accelerator.lock");
    let first_policies = Rc::new(RefCell::new(Vec::new()));
    let second_policies = Rc::new(RefCell::new(Vec::new()));

    let first = BrokerEngine::load_with(acquire(&lock_path).into(), {
        let policies = Rc::clone(&first_policies);
        move |policy| {
            policies.borrow_mut().push(policy);
            Ok(FakeEngine::new(
                "metal",
                true,
                [Outcome::Success],
                Rc::new(Cell::new(0)),
            ))
        }
    })
    .expect("first model loads");

    let second_lease =
        AcceleratorLease::try_acquire(&lock_path).expect("second lock attempt is usable");
    assert!(second_lease.is_none());
    let second = BrokerEngine::load_with(second_lease, {
        let policies = Rc::clone(&second_policies);
        move |policy| {
            policies.borrow_mut().push(policy);
            Ok(FakeEngine::new(
                "cpu",
                false,
                [Outcome::Success],
                Rc::new(Cell::new(0)),
            ))
        }
    })
    .expect("second model loads on CPU");

    assert_eq!(first_policies.borrow().as_slice(), [BackendPolicy::Auto]);
    assert_eq!(
        second_policies.borrow().as_slice(),
        [BackendPolicy::CpuOnly]
    );
    assert!(first.accelerator_lease_held());
    assert!(!second.accelerator_lease_held());
}

#[test]
fn broker_accelerator_resolved_cpu_or_unready_engine_releases_an_acquired_lease() {
    for backend in ["cpu", "unready"] {
        let temp = tempfile::tempdir().expect("tempdir");
        let lock_path = temp.path().join("accelerator.lock");
        let engine = BrokerEngine::load_with(Some(acquire(&lock_path)), move |policy| {
            assert_eq!(policy, BackendPolicy::Auto);
            Ok(FakeEngine::new(
                backend,
                false,
                [Outcome::Success],
                Rc::new(Cell::new(0)),
            ))
        })
        .expect("non-accelerated engine loads");

        assert!(!engine.accelerator_lease_held());
        drop(acquire(&lock_path));
    }
}

#[test]
fn broker_accelerator_cpu_resource_exhaustion_does_not_retry() {
    let policies = Rc::new(RefCell::new(Vec::new()));
    let calls = Rc::new(Cell::new(0));
    let engine = BrokerEngine::load_with(None, {
        let policies = Rc::clone(&policies);
        let calls = Rc::clone(&calls);
        move |policy| {
            policies.borrow_mut().push(policy);
            Ok(FakeEngine::new(
                "cpu",
                false,
                [Outcome::ResourceExhausted],
                Rc::clone(&calls),
            ))
        }
    })
    .expect("CPU engine loads");

    let err = engine
        .embed(&["query".to_string()], Role::Query)
        .expect_err("CPU exhaustion is returned");
    assert_eq!(err.failure_class, EngineFailureClass::ResourceExhausted);
    assert_eq!(calls.get(), 1);
    assert_eq!(policies.borrow().as_slice(), [BackendPolicy::CpuOnly]);
}

#[test]
fn broker_accelerator_static_engine_without_recovery_loader_returns_the_original_failure() {
    let temp = tempfile::tempdir().expect("tempdir");
    let lock_path = temp.path().join("accelerator.lock");
    let calls = Rc::new(Cell::new(0));
    let engine = BrokerEngine::new(
        FakeEngine::new(
            "metal",
            true,
            [Outcome::ResourceExhausted],
            Rc::clone(&calls),
        ),
        Some(acquire(&lock_path)),
    );

    let err = engine
        .embed(&["query".to_string()], Role::Query)
        .expect_err("static engine cannot recover");

    assert_eq!(err.failure_class, EngineFailureClass::ResourceExhausted);
    assert_eq!(calls.get(), 1);
    assert!(engine.accelerator_lease_held());
}

#[test]
fn broker_accelerator_ordinary_failures_never_demote_or_retry() {
    for kind in ["Decode", "Encode", "Item", "Application"] {
        let temp = tempfile::tempdir().expect("tempdir");
        let lock_path = temp.path().join("accelerator.lock");
        let policies = Rc::new(RefCell::new(Vec::new()));
        let calls = Rc::new(Cell::new(0));
        let engine = BrokerEngine::load_with(Some(acquire(&lock_path)), {
            let policies = Rc::clone(&policies);
            let calls = Rc::clone(&calls);
            move |policy| {
                policies.borrow_mut().push(policy);
                match policy {
                    BackendPolicy::Auto => Ok(FakeEngine::new(
                        "metal",
                        true,
                        [Outcome::Application(kind)],
                        Rc::clone(&calls),
                    )),
                    BackendPolicy::CpuOnly => panic!("ordinary failure must not load CPU"),
                }
            }
        })
        .expect("accelerated engine loads");

        let err = engine
            .embed(&["query".to_string()], Role::Query)
            .expect_err("ordinary failure is returned");
        assert_eq!(err.kind, kind);
        assert_eq!(err.failure_class, EngineFailureClass::Application);
        assert_eq!(calls.get(), 1);
        assert_eq!(policies.borrow().as_slice(), [BackendPolicy::Auto]);
        assert!(engine.accelerator_lease_held());
    }
}

#[test]
fn broker_accelerator_health_reports_resolved_cpu_degradation_and_released_lease() {
    let temp = tempfile::tempdir().expect("tempdir");
    let lock_path = temp.path().join("accelerator.lock");
    let engine = BrokerEngine::load_with(acquire(&lock_path).into(), move |policy| {
        Ok(match policy {
            BackendPolicy::Auto => FakeEngine::new(
                "metal",
                true,
                [Outcome::ResourceExhausted],
                Rc::new(Cell::new(0)),
            ),
            BackendPolicy::CpuOnly => {
                FakeEngine::new("cpu", false, [Outcome::Success], Rc::new(Cell::new(0)))
            }
        })
    })
    .expect("accelerated engine loads");

    engine
        .embed(&["query".to_string()], Role::Query)
        .expect("request retries");
    let health = engine.health_facts().expect("health succeeds");

    assert_eq!(health["resolved_backend"], "cpu");
    assert_eq!(health["accelerated"], false);
    assert_eq!(health["accelerator_lease_held"], false);
    assert_eq!(health["degraded_reason"], RUNTIME_DEGRADATION);
    assert_eq!(
        health["load_policy"]["degraded_reason"],
        RUNTIME_DEGRADATION
    );
}
