use axum::response::IntoResponse;
use thiserror::Error;
use tokio::io::Error as TokioIoError;
use tokio::net::{TcpListener, ToSocketAddrs};
use utoipa::ToSchema;
use utoipa_axum::router::OpenApiRouter;
use utoipa_axum::routes;

mod audio;

/// Define all the possible errors for OpenAI Compatible Endpoint
#[derive(Debug, Error)]
pub enum OpenAiError {
    #[error("I/O Error occured: {0}")]
    IoError(#[from] TokioIoError),
}

pub struct OpenAi {
    router: OpenApiRouter,
}

impl OpenAi {
    pub fn new() -> Self {
        Self {
            router: OpenApiRouter::new(),
        }
    }

    ///
    pub fn with_create_transcriptions(mut self) -> Self {
        self.router = self
            .router
            .routes(routes!(audio::transcription::create_transcription));

        self
    }

    pub async fn listen<A: ToSocketAddrs>(self, interface: A) {
        let transport = TcpListener::bind(interface).await.expect("Failed to bind");
        let (router, api) = OpenApiRouter::new()
            .nest("/api/v1", self.router)
            .split_for_parts();

        axum::serve(transport, router)
            .await
            .expect("Failed to serve");
    }
}
