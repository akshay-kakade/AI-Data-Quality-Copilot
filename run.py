"""
Entry point for the AI Data Quality Copilot.
Mounts the Gradio UI onto the FastAPI server.
"""
import gradio as gr
from backend.main import app
from frontend.app import create_gradio_app

# Create the Gradio app
gradio_app = create_gradio_app()

# Mount Gradio onto FastAPI at root path
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=8080, reload=True)

