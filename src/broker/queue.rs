use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::time::Instant;

const INTERACTIVE_BURST: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestClass {
    Interactive,
    Batch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    Full,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Dequeued<T> {
    Ready(T),
    Expired(T),
}

pub struct BrokerQueue<T> {
    capacity: usize,
    state: Mutex<State<T>>,
    ready: Condvar,
}

struct State<T> {
    interactive: VecDeque<Queued<T>>,
    batch: VecDeque<Queued<T>>,
    interactive_since_batch_waited: usize,
}

struct Queued<T> {
    value: T,
    deadline: Instant,
}

impl<T> BrokerQueue<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            state: Mutex::new(State {
                interactive: VecDeque::new(),
                batch: VecDeque::new(),
                interactive_since_batch_waited: 0,
            }),
            ready: Condvar::new(),
        }
    }

    pub fn try_enqueue(
        &self,
        class: RequestClass,
        value: T,
        deadline: Instant,
    ) -> Result<(), QueueError> {
        let mut state = self.state.lock().expect("broker queue mutex poisoned");
        if state.interactive.len() + state.batch.len() >= self.capacity {
            return Err(QueueError::Full);
        }
        let queued = Queued { value, deadline };
        match class {
            RequestClass::Interactive => state.interactive.push_back(queued),
            RequestClass::Batch => state.batch.push_back(queued),
        }
        self.ready.notify_one();
        Ok(())
    }

    pub fn dequeue(&self) -> Dequeued<T> {
        let mut state = self.state.lock().expect("broker queue mutex poisoned");
        loop {
            if let Some(queued) = take_next(&mut state) {
                return if queued.deadline <= Instant::now() {
                    Dequeued::Expired(queued.value)
                } else {
                    Dequeued::Ready(queued.value)
                };
            }
            state = self.ready.wait(state).expect("broker queue mutex poisoned");
        }
    }
}

fn take_next<T>(state: &mut State<T>) -> Option<Queued<T>> {
    if !state.batch.is_empty() {
        if !state.interactive.is_empty() && state.interactive_since_batch_waited < INTERACTIVE_BURST
        {
            state.interactive_since_batch_waited += 1;
            return state.interactive.pop_front();
        }
        state.interactive_since_batch_waited = 0;
        return state.batch.pop_front();
    }
    state.interactive_since_batch_waited = 0;
    state.interactive.pop_front()
}
