use std::{sync::Arc, time::Instant};

use cgmath::Vector2;
use log::{error, info};
use palette::{Srgba, rgb::channels::Rgba};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::ActiveEventLoop,
    window::{Window, WindowId},
};

use crate::renderer::Renderer;

struct State {
    last_update: Instant,
}

#[derive(Default)]
pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = match event_loop.create_window(Window::default_attributes()) {
            Ok(window) => Arc::new(window),
            Err(err) => {
                return error!("Failed to create window: {}", err);
            }
        };

        match pollster::block_on(Renderer::new(window.clone())) {
            Ok(renderer) => {
                self.renderer = Some(renderer);
                self.state = Some(State {
                    last_update: Instant::now(),
                });
            }
            Err(err) => {
                error!("Failed to create renderer: {}", err);
            }
        }

        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        info!("Window event! {:?}", event);
        let Some(window) = self.window.as_ref() else {
            return info!("Skip window_event handling. We have no window");
        };

        let Some(renderer) = self.renderer.as_mut() else {
            return info!("Skip window_event handling. We have no renderer");
        };

        let Some(state) = self.state.as_mut() else {
            return info!("Skip window_event handling. We have no state");
        };

        match event {
            WindowEvent::CloseRequested => {
                info!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let _delta = now.duration_since(state.last_update).as_secs_f32();
                state.last_update = now;

                // Render:
                {
                    renderer.begin_drawing();
                    renderer.clear_color(Srgba::new(24. / 255., 24. / 255., 27. / 255., 1.));

                    renderer.draw_text("Hello Renderer", Vector2::new(0., 0.), 22., 22., None);

                    // TODO: draw here…
                    if let Err(err) = renderer.end_drawing() {
                        error!("Error: renderer.render(): {}", err);
                    }
                }

                window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                renderer.resize(size);
            }
            _ => (),
        }
    }
}
