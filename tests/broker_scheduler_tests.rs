use julie_semantic_sidecar::broker::queue::{BrokerQueue, Dequeued, QueueError, RequestClass};
use std::time::{Duration, Instant};

#[test]
fn waiting_batch_runs_after_at_most_eight_interactive_dequeues() {
    let queue = BrokerQueue::new(64);
    queue
        .try_enqueue(
            RequestClass::Batch,
            99,
            Instant::now() + Duration::from_secs(1),
        )
        .unwrap();
    for interactive in 0..12 {
        queue
            .try_enqueue(
                RequestClass::Interactive,
                interactive,
                Instant::now() + Duration::from_secs(1),
            )
            .unwrap();
    }

    let first: Vec<_> = (0..8)
        .map(|_| queue.dequeue())
        .map(|item| match item {
            Dequeued::Ready(item) => item,
            Dequeued::Expired(_) => panic!("interactive work expired"),
        })
        .collect();
    assert_eq!(first, (0..8).collect::<Vec<_>>());
    assert_eq!(queue.dequeue(), Dequeued::Ready(99));
}

#[test]
fn queue_rejects_the_sixty_fifth_waiting_request() {
    let queue = BrokerQueue::new(64);
    for request in 0..64 {
        queue
            .try_enqueue(
                RequestClass::Interactive,
                request,
                Instant::now() + Duration::from_secs(1),
            )
            .unwrap();
    }

    assert_eq!(
        queue.try_enqueue(
            RequestClass::Interactive,
            64,
            Instant::now() + Duration::from_secs(1),
        ),
        Err(QueueError::Full)
    );
}

#[test]
fn an_expired_request_is_returned_without_entering_the_engine() {
    let queue = BrokerQueue::new(64);
    queue
        .try_enqueue(
            RequestClass::Interactive,
            "expired",
            Instant::now() - Duration::from_millis(1),
        )
        .unwrap();

    assert_eq!(queue.dequeue(), Dequeued::Expired("expired"));
}
