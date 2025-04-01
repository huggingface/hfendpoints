use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinError;

///
pub trait Handler {
    type Request;
    type Response;

    ///
    ///
    /// # Arguments
    ///
    /// * `request`:
    ///
    /// returns: Self::Response
    ///
    /// # Examples
    ///
    /// ```
    ///
    /// ```
    fn on_request(&self, request: Self::Request) -> Self::Response;
}

pub async fn spawn_handler<I: Send + 'static, O: Send + 'static>(
    mut receiver: Receiver<(I, Sender<O>)>,
) -> Result<(), JoinError> {
    tokio::spawn(async move {
        'outer: loop {
            if let Some((request, responses_channel)) = receiver.recv().await {
                println!("Received request");
            } else {
                break 'outer;
            }
        }
    })
    .await
}
