use crate::{AUDIO_DESC, AUDIO_TAG, EMBEDDINGS_DESC, EMBEDDINGS_TAG, STATUS_DESC, STATUS_TAG};
use utoipa::OpenApi;

#[derive(OpenApi)]
#[openapi(
    info(title = "🤗 Inference Endpoint API Specifications"),
    tags(
        (name = STATUS_TAG, description = STATUS_DESC),
        (name = AUDIO_TAG, description = AUDIO_DESC),
        (name = EMBEDDINGS_TAG, description = EMBEDDINGS_DESC)
    )
)]
pub struct ApiDoc;
