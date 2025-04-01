use crate::Error;
use log::{error, info};
use std::sync::Arc;
use std::thread::{spawn, JoinHandle};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

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
    fn on_request(&self, request: Self::Request) -> Result<Self::Response, Error>;
}

pub fn spawn_handler<I, O, H>(
    mut ingress: UnboundedReceiver<(I, UnboundedSender<Result<O, Error>>)>,
    background_handler: Arc<H>,
) -> JoinHandle<()>
where
    I: Send + 'static,
    O: Send + 'static,
    H: Handler<Request = I, Response = O> + Send + Sync + 'static,
{
    spawn(move || {
        loop {
            if let Some((request, egress)) = ingress.blocking_recv() {
                info!("[LOOPER] Received request");
                let response = background_handler.on_request(request);
                info!("[LOOPER] Response ready");
                if let Err(e) = egress.send(response) {
                    error!("Failed to send back response to client: {e}");
                }
            }
        }
    })
}
