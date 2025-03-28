// /// Endpoint is the high-level representation of what's running under the Hugging Face Inference Endpoint Platform.
// /// It describes the way to create the underlying inference engine, along with
// /// the kind of I/O types exchanged between the API and the inference engine.
// pub trait Endpoint {
//     type Error;
//     type Factory: HandlerFactory;
//
//     /// Factory method to create the underlying `Handler` to process incoming request.
//     /// This method will be called from a background thread, managed by hfendpoints to make
//     /// sure we have complete isolation from the API and the inference engine.
//     fn handler(
//         &self,
//     ) -> impl Handler<crate::Request=Self::Factory::Request, crate::Response=Self::Factory::Response>;
//
//     /// Run the actual endpoint
//     fn run<A: ToSocketAddrs>(
//         &self,
//         interface: A,
//     ) -> impl Future<Output=Result<(), Self::Error>> + Send;
// }
