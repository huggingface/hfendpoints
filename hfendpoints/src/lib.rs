#[cfg(feature = "python")]
mod python;
mod openai;

/// High-level object containing all the information required to run the actual endpoint
pub trait Endpoint {
    /// Main entrypoint for running the actual endpoint
    async fn run(&self);
}
