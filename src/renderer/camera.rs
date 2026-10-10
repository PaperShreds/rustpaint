use glam::{Mat4, Vec3};

/// Orbits around a target point.
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f32,
    /// Approximate size of the thing we're looking at; limits zoom and sets clip planes.
    pub radius: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 3.0,
            radius: 1.0,
            yaw: 0.6,
            pitch: 0.4,
            fov_y: 45.0_f32.to_radians(),
        }
    }
}

impl OrbitCamera {
    pub fn eye(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        self.target + self.distance * Vec3::new(cos_pitch * sin_yaw, sin_pitch, cos_pitch * cos_yaw)
    }

    /// Point the camera at a bounding sphere so it fills the view nicely.
    pub fn frame(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        self.radius = radius;
        self.distance = radius * 3.0;
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        let near = self.radius * 0.01;
        let far = self.radius * 100.0;
        let view = glam::camera::rh::view::look_at_mat4(self.eye(), self.target, Vec3::Y);
        let proj = glam::camera::rh::proj::directx::perspective(self.fov_y, aspect, near, far);
        proj * view
    }

    /// `dx` / `dy` are mouse movement in points. Flip a sign if a direction feels backwards.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.01;
        self.pitch = (self.pitch + dy * 0.01).clamp(-1.5, 1.5);
    }

    pub fn zoom(&mut self, scroll: f32) {
        self.distance =
            (self.distance * (-scroll * 0.002).exp()).clamp(self.radius * 0.2, self.radius * 20.0);
    }
}
