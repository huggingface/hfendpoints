use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// Store some information about the context in which the endpoint runs
#[derive(Clone)]
pub struct EndpointContext<I, O> {
    // /// Realtime information streaming about underlying resources usage of the handler
    // in_flight_tracker: Receiver<InFlightStats>,
    ipc: UnboundedSender<(I, UnboundedSender<O>)>,
}

impl<I, O> EndpointContext<I, O> {
    pub fn new(ipc: UnboundedSender<(I, UnboundedSender<O>)>) -> Self {
        Self { ipc }
    }

    pub fn schedule(&self, request: I) -> UnboundedReceiver<O> {
        let (sender, receiver) = unbounded_channel();
        self.ipc
            .send((request, sender))
            .expect("Failed to send request");

        receiver
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
