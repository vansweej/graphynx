use thiserror::Error;

// ── DeviceIdError ─────────────────────────────────────────────────────────────

/// Errors produced when constructing a [`DeviceId`] through a safe constructor.
#[derive(Debug, Error, Clone, Eq, PartialEq)]
pub enum DeviceIdError {
    /// A device identifier was empty. Device IDs must be non-empty so they
    /// can meaningfully identify a backend instance.
    #[error("Device ID must not be empty")]
    Empty,
}

// ── DeviceId ──────────────────────────────────────────────────────────────────

/// Identifies a backend instance at runtime.
///
/// By convention, use the form `"<backend>:<index>"` for hardware backends
/// and `"<runtime>:<device>"` for ML runtime backends.
///
/// # Examples
/// `"cpu"`, `"cuda:0"`, `"opencl:1"`, `"onnx:cpu"`, `"libtorch:cuda:0"`
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct DeviceId(String);

impl DeviceId {
    /// Construct a `DeviceId` from any non-empty string.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceIdError::Empty`] if the identifier string is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use graph_core::types::device_id::DeviceId;
    ///
    /// assert!(DeviceId::try_new("cuda:0").is_ok());
    /// assert!(DeviceId::try_new("").is_err());
    /// ```
    pub fn try_new(id: impl Into<String>) -> Result<Self, DeviceIdError> {
        let s = id.into();
        if s.is_empty() {
            Err(DeviceIdError::Empty)
        } else {
            Ok(Self(s))
        }
    }

    /// Construct a `DeviceId` from any string without validation.
    ///
    /// Prefer [`DeviceId::try_new`] in production code. This constructor
    /// exists for convenience in tests and known-good literals.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Returns the device identifier as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for DeviceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    #[test]
    fn device_id_new_from_str() {
        let id = DeviceId::new("cuda:0");
        assert_eq!(id.as_str(), "cuda:0");
    }

    #[test]
    fn device_id_new_from_string() {
        let id = DeviceId::new(String::from("opencl:1"));
        assert_eq!(id.as_str(), "opencl:1");
    }

    #[test]
    fn device_id_display() {
        let id = DeviceId::new("cuda:0");
        assert_eq!(format!("{id}"), "cuda:0");
    }

    #[test]
    fn device_id_debug() {
        let id = DeviceId::new("cpu");
        let debug = format!("{id:?}");
        assert!(debug.contains("cpu"));
    }

    #[test]
    fn device_id_clone() {
        let id = DeviceId::new("cuda:0");
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn device_id_equality() {
        let a = DeviceId::new("cuda:0");
        let b = DeviceId::new("cuda:0");
        let c = DeviceId::new("cuda:1");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn device_id_hash_as_map_key() {
        let mut map: HashMap<DeviceId, &str> = HashMap::new();
        map.insert(DeviceId::new("cuda:0"), "gpu0");
        map.insert(DeviceId::new("cpu"), "host");

        assert_eq!(map.get(&DeviceId::new("cuda:0")), Some(&"gpu0"));
        assert_eq!(map.get(&DeviceId::new("cpu")), Some(&"host"));
        assert_eq!(map.get(&DeviceId::new("cuda:1")), None);
    }

    #[test]
    fn device_id_empty_string() {
        let id = DeviceId::new("");
        assert_eq!(format!("{id}"), "");
    }

    #[test]
    fn device_id_try_new_valid() {
        let id = DeviceId::try_new("cuda:0").unwrap();
        assert_eq!(id.as_str(), "cuda:0");
    }

    #[test]
    fn device_id_try_new_empty_is_error() {
        assert_eq!(DeviceId::try_new(""), Err(DeviceIdError::Empty));
    }

    #[test]
    fn device_id_try_new_from_string() {
        let id = DeviceId::try_new(String::from("opencl:1")).unwrap();
        assert_eq!(id.as_str(), "opencl:1");
    }

    #[test]
    fn device_id_as_str() {
        let id = DeviceId::new("cpu");
        assert_eq!(id.as_str(), "cpu");
    }

    #[test]
    fn device_id_error_display() {
        let err = DeviceIdError::Empty;
        assert!(err.to_string().to_lowercase().contains("empty"));
    }

    #[test]
    fn device_id_error_clone_eq() {
        let err = DeviceIdError::Empty;
        assert_eq!(err.clone(), err);
    }
}
