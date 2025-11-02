import numpy as np
import tensorflow as tf
from PIL import Image
import gradio as gr

model = tf.keras.models.load_model("generator.keras")


def generate_face(smiling, eyeglasses, male, seed, size):
    np.random.seed(int(seed))
    z = np.random.normal(size=(1, 100)).astype(np.float32)
    attrs = np.array([[smiling, eyeglasses, male]], dtype=np.float32)

    img = model.predict([z, attrs], verbose=0)
    img = np.clip((img[0] + 1.0) * 0.5, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    if size != 64:
        pil_img = pil_img.resize((size, size), Image.NEAREST)

    return pil_img


demo = gr.Interface(
    fn=generate_face,
    inputs=[
        gr.Radio([0, 1], value=0, label="Smiling", info="0=No, 1=Yes"),
        gr.Radio([0, 1], value=0, label="Eyeglasses", info="0=No, 1=Yes"),
        gr.Radio([0, 1], value=0, label="Gender", info="0=Female, 1=Male"),
        gr.Number(value=42, label="Seed"),
        gr.Radio([64, 128, 256, 512], value=256, label="Output Size"),
    ],
    outputs=gr.Image(type="pil", label="Generated Face", show_download_button=True),
    title="Conditional Face Generator",
    description="Generate faces with controllable attributes",
    examples=[
        [1, 0, 0, 42, 256],
        [0, 1, 1, 123, 512],
        [1, 1, 0, 456, 128],
        [0, 0, 1, 789, 64],
    ],
)

demo.launch()
