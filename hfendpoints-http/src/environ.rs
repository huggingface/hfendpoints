use hfendpoints_core::environ::{EnvironmentError, TryFromEnv};
use std::str::FromStr;
use std::time::Duration;
use tower_http::timeout::TimeoutLayer;
use tracing::debug;

/// Represents a timeout configuration with a specified duration.
///
/// The `Timeout` struct holds a duration of time that can be used to
/// represent or configure timeouts in an application.
///
/// # Attributes
/// * `duration` - A `Duration` instance representing the length of the timeout.
///
/// # Debugging
/// When compiled in debug mode (with `debug_assertions` enabled), this
/// struct derives the `Debug` trait, allowing it to be printed using the `{:?}`
/// formatter for debugging purposes.
///
/// # Examples
///
/// ```
/// use std::time::Duration;
/// use hfendpoints_http::environ::Timeout;
///
/// let timeout = Timeout {
///     duration: Duration::new(5, 0), // 5 seconds
/// };
///
/// // Use the timeout in your application...
/// ```
#[cfg_attr(debug_assertions, derive(Debug))]
pub struct Timeout {
    pub duration: Duration,
}

impl TryFromEnv for Timeout {
    /// A constant representing the name of the environment variable used to configure
    /// the timeout (in seconds) for requests made to HF Endpoints.
    ///
    /// The value of this environment variable, if set, will determine the request
    /// timeout duration. If the environment variable is not set, a default timeout may be used.
    ///
    /// # Example:
    /// ```
    /// use std::env;
    /// use hfendpoints_core::environ::TryFromEnv;
    /// use hfendpoints_http::environ::Timeout;
    ///
    /// // Set the environment variable
    /// unsafe { env::set_var(Timeout::ENV_VAR_NAME, "10"); }
    ///
    /// // Retrieve the timeout duration from the environment variable
    /// let timeout = env::var(Timeout::ENV_VAR_NAME).unwrap_or_else(|_| "5".to_string());
    /// assert_eq!(timeout, "10");
    /// ```
    ///
    /// # Constant:
    /// - `ENV_VAR_NAME`: The name/key of the environment variable.
    const ENV_VAR_NAME: &'static str = "HFENDPOINTS_REQUEST_TIMEOUT_SEC";

    /// Attempts to create an instance of the implementing type `Self` by retrieving
    /// a value from an environment variable. The environment variable name is defined
    /// by the associated constant `ENV_VAR_NAME`.
    ///
    /// # Behavior
    /// - If the environment variable defined by `ENV_VAR_NAME` is present, it attempts
    ///   to parse its value as a 64-bit unsigned integer (`u64`), representing a
    ///   timeout in seconds.
    /// - If the environment variable is absent, a default value of `120` seconds is used.
    /// - If parsing the environment variable value fails, an error is returned.
    ///
    /// # Returns
    /// - `Ok(Self)`: If the environment variable value or default value is successfully
    ///   parsed and converted into the appropriate type.
    /// - `Err(EnvironmentError)`: If the conversion of the environment variable value
    ///   to `u64` fails, an error of type `EnvironmentError::InvalidEnvVar` is returned.
    ///
    /// # Type Parameters
    /// - `Self: Sized`: The method is generic over `Self` and requires it to be a
    ///   sized type. The method assumes `Self` has a structure that includes a
    ///   `duration` field of the type `std::time::Duration`.
    ///
    /// # Logging
    /// - Logs a debug message indicating the timeout value being set, whether it
    ///   was retrieved from the environment or derived from the default value.
    ///
    /// # Errors
    /// - `EnvironmentError::InvalidEnvVar`: Returned when the value of the environment
    ///   variable cannot be parsed into a `u64`.
    ///
    /// # Example
    /// ```rust
    /// use hfendpoints_core::environ::TryFromEnv;
    /// use hfendpoints_http::environ::Timeout;
    ///
    /// let timeout = Timeout::try_from_env();
    /// match timeout {
    ///     Ok(timeout) => println!("Timeout set to: {:?}", timeout),
    ///     Err(e) => eprintln!("Failed to parse environment variable: {:?}", e),
    /// }
    /// ```
    ///
    /// # Notes
    /// - This function uses `std::env::var` to fetch the environment variable's value,
    ///   which inherits its limitations (e.g., environment variable name must exist in
    ///   the current process's environment).
    fn try_from_env() -> Result<Self, EnvironmentError>
    where
        Self: Sized,
    {
        match u64::from_str(&std::env::var(Self::ENV_VAR_NAME).unwrap_or(String::from("120"))) {
            Ok(timeout) => {
                debug!("[Environ] Timeout set to {} seconds", timeout);
                Ok(Self {
                    duration: Duration::from_secs(timeout),
                })
            }
            Err(err) => Err(EnvironmentError::InvalidEnvVar(
                Self::ENV_VAR_NAME.into(),
                err.to_string(),
            )),
        }
    }
}

/// Helper trait implementation to convert an environment extracted Timeout to `tower_http::timeout::service::TimeoutLayer`
impl Into<TimeoutLayer> for Timeout {
    fn into(self) -> TimeoutLayer {
        TimeoutLayer::new(self.duration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_timeout_with_valid_small_integer() {
        unsafe {
            env::set_var(Timeout::ENV_VAR_NAME, "5");
        }
        let timeout = Timeout::try_from_env().unwrap();
        assert_eq!(timeout.duration, Duration::from_secs(5));
    }

    #[test]
    fn test_timeout_with_invalid_string() {
        unsafe {
            env::set_var(Timeout::ENV_VAR_NAME, "invalid");
        }
        let result = Timeout::try_from_env();
        assert!(matches!(result, Err(EnvironmentError::InvalidEnvVar(_, _))));
    }
}
