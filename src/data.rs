use serde_json::{Map, Value};

pub type JsonMap = Map<String, Value>;
pub type JsonValue = Value;

#[macro_export]
macro_rules! json_map {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut map = $crate::prelude::JsonMap::new();
            $(
                map.insert($key.to_string(), serde_json::json!($value));
            )*
            map
        }
    };
}
