use crate::Error;
use std::sync::Arc;
use tokio::spawn;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{debug, error, span, warn, Instrument, Level};

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
    fn on_request(
        &self,
        request: Self::Request,
    ) -> impl Future<Output=Result<Self::Response, Error>> + Send;
}

pub async fn wait_for_requests<I, O, H>(
    mut ingress: UnboundedReceiver<(I, UnboundedSender<Result<O, Error>>)>,
    background_handler: Arc<H>,
) where
    I: Send + 'static,
    O: Send + 'static,
    H: Handler<Request=I, Response=O> + Send + Sync + 'static,
{
    'looper: loop {
        if let Some((request, egress)) = ingress.recv().await {
            debug!("[LOOPER] Received request");
            let background_handler = Arc::clone(&background_handler);
            let sp_on_request = span!(Level::DEBUG, "on_request");

            spawn(
                async move {
                    let response = background_handler.on_request(request).await;
                    if let Err(e) = egress.send(response) {
                        error!("Failed to send back response to client: {e}");
                    }
                }.instrument(sp_on_request),
            );
        } else {
            warn!("[LOOPER] received a termination notice from ingress channel, exiting");
            break 'looper;
        }
    }
}
