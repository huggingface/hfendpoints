pub mod openai;

pub trait EndpointRouter {
    type Request;
    type Response;
}
