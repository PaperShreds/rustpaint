pub struct Brush {
    pub size: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub color: [f32; 3],
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            size: 32.0,
            hardness: 0.8,
            opacity: 1.0,
            color: [1.0, 1.0, 1.0],
        }
    }
}
