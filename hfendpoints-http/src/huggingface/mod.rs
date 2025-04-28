use serde::Deserialize;
use std::fmt::Debug;
use utoipa::ToSchema;

pub(crate) mod asr;

/// Generic representation of requests sent from Hugging Face inference definition.
///
#[cfg_attr(debug_assertions, derive(Debug))]
#[derive(Clone, Deserialize, ToSchema)]
pub struct HuggingFaceRequest<I, P>
where
    I: Debug + ToSchema,
    P: Debug + ToSchema,
{
    inputs: I,
    parameters: P,
}
