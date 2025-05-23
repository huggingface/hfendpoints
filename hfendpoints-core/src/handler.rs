use crate::Error;
use std::borrow::Cow;
use std::sync::Arc;
use tokio::spawn;
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};
use tracing::{debug, error, span, warn, Instrument, Level};

#[derive(Clone, Debug, Error)]
pub enum HandlerError {
    #[error("Failed to send message through IPC: {0}")]
    IpcFailed(Cow<'static, str>),

    #[cfg(feature = "python")]
    #[error("Python handler implementation is not correct: {0}")]
    Implementation(Cow<'static, str>),
}

/// A trait that represents a generic handler for processing requests asynchronously.
///
/// This trait defines how to handle a specific `Request` type and produce a `Response` type
/// using asynchronous operations.
/// It is commonly used in scenarios
/// where a request-response mechanism is required with support for asynchronous execution,
/// such as in servers or middleware systems.
///
/// # Associated Types
///
/// * `Request`: The type of the incoming request that this handler processes.
/// * `Response`: The type of the outgoing response produced by the handler.
///
/// # Required Methods
///
/// ## `on_request`
///
/// Handles the incoming request and returns a `Future` that resolves to a `Result` containing
/// either the processed `Response` or an `Error` if the operation fails.
///
/// # Arguments
///
/// - `request`: The incoming request of type `Self::Request` that needs to be handled.
///
/// # Returns
///
/// Returns a future that resolves to a `Result` where:
/// - `Ok(Self::Response)`: Indicates successful processing of the request.
/// - `Err(Error)`: Indicates an error occurred during processing.
///
/// The future must be `Send`, making it suitable for use in multithreaded contexts.
///
/// # Examples
///
/// ```
/// use hfendpoints_core::{EndpointResult, Handler};
/// use std::future::Future;
/// use std::pin::Pin;
/// use std::task::{Context, Poll};
///
/// struct MyHandler;
///
/// impl Handler for MyHandler {
///     type Request = String;
///     type Response = String;
///
///     fn on_request(
///         &self,
///         request: Self::Request,
///     ) -> impl Future<Output = EndpointResult<Self::Response>> + Send {
///         async move {
///             // Process the request and return the response.
///             Ok(format!("Processed: {}", request))
///         }
///     }
/// }
///
/// // Example usage
/// #[tokio::main]
/// async fn main() {
///     let handler = MyHandler;
///     let response = handler.on_request("Test request".to_string()).await;
///     match response {
///         Ok(res) => println!("Response: {}", res),
///         Err(err) => println!("Error: {:?}", err),
///     }
/// }
/// ```
///
/// Note: In the example, replace `Error` with a concrete error type that suits your use case.
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
    ) -> impl Future<Output = Result<Self::Response, Error>> + Send;
}

pub async fn wait_for_requests<I, O, H>(
    mut ingress: UnboundedReceiver<(I, UnboundedSender<Result<O, Error>>)>,
    background_handler: Arc<H>,
) where
    I: Send + 'static,
    O: Send + 'static,
    H: Handler<Request = I, Response = O> + Send + Sync + 'static,
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
                }
                .instrument(sp_on_request),
            );
        } else {
            warn!("[LOOPER] received a termination notice from ingress channel, exiting");
            break 'looper;
        }
    }
}
