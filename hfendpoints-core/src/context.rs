use crate::handler::HandlerError::IpcFailed;
use crate::{EndpointResult, Error};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// Store some information about the context in which the endpoint runs
///
/// Cloning this context will only clone the underlying file descriptor used for the IPC channel
#[derive(Clone)]
pub struct EndpointContext<I, O> {
    // /// Realtime information streaming about underlying resources usage of the handler
    // in_flight_tracker: Receiver<InFlightStats>,
    ipc: UnboundedSender<(I, UnboundedSender<EndpointResult<O>>)>,
}

impl<I, O> EndpointContext<I, O> {
    pub fn new(ipc: UnboundedSender<(I, UnboundedSender<EndpointResult<O>>)>) -> Self {
        Self { ipc }
    }

    pub fn schedule(&self, request: I) -> EndpointResult<UnboundedReceiver<EndpointResult<O>>> {
        let (sender, receiver) = unbounded_channel();
        if let Err(e) = self.ipc.send((request, sender)) {
            return Err(Error::Handler(IpcFailed(e.to_string().into())));
        }

        Ok(receiver)
    }

    // ///
    // ///
    // /// # Arguments
    // ///
    // /// * `name`:
    // ///
    // /// returns: ()
    // ///
    // /// # Examples
    // ///
    // /// ```
    // ///
    // /// ```
    // pub fn register_custom_metric(&mut self, name: &'static str) {}
    //
    // ///
    // ///
    // /// # Arguments
    // ///
    // /// * `stats`:
    // ///
    // /// returns: ()
    // ///
    // /// # Examples
    // ///
    // /// ```
    // ///
    // /// ```
    // pub fn push_in_flight_stats(&self, stats: InFlightStats) {}
}
