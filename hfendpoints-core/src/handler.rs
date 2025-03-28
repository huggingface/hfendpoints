pub trait HandlerFactory {
    type Error;

    /// Express the type for incoming requests getting through the endpoint
    type Request;

    /// Express the type for outgoing responses from the endpoint
    type Response;

    fn create(&self) -> impl Handler<Request = Self::Request, Response = Self::Response>;
}

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
    fn on_request(&mut self, request: Self::Request)
    -> impl Future<Output = Self::Response> + Send;
}

// pub async fn spawn_handler<H: HandlerFactory>(
//     factory: H,
//     receiver: Receiver<String>,
// ) -> Result<(), JoinError> {
//     tokio::spawn(async || {
//         let handler = factory.create();
//     })
//     .await
// }
