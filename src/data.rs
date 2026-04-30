use serde_json::Value;

pub type JsonValue = Value;
pub type StringMap = std::collections::HashMap<String, String>;

#[macro_export]
macro_rules! string_map {
    ($($key:expr => $value:expr),* $(,)?) => {
        {
            let mut map = std::collections::HashMap::new();
            $(
                map.insert($key.to_string(), $value.to_string());
            )*
            map

        }
    };
}
