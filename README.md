<h1 align="center">Hebrew-Mistral-7B-GUI</h1>

<h3 align="center">GUI for Generating High-Quality Hebrew Text with Hebrew-Mistral-7B</h3>

<p align="left">
A user-friendly interface for interacting with the powerful Hebrew-Mistral-7B language model, allowing you to generate coherent and contextually relevant text in Hebrew with adjustable parameters for customized output.
</p>

![Yam Peleg](https://github.com/ShmuelRonen/Hebrew-Mistral-7B-GUI/assets/80190186/e5fb5ef8-e7c0-48c2-9354-01522d76a3d0)



## Installation

### One-click Installation

1. Clone the repository:
```
git clone https://github.com/your-username/Hebrew-Mistral-7B-GUI.git
cd Hebrew-Mistral-7B-GUI
```

2. Run the:
```
init_env.bat
```
script to automatically set up the virtual environment and install the required dependencies.

### Manual Installation

1. Clone the repository:
```
git clone https://github.com/your-username/Hebrew-Mistral-7B-GUI.git
cd Hebrew-Mistral-7B-GUI
Copy code
3. Create and activate a virtual environment:
python -m venv venv
venv\Scripts\activate
```

4. Install the required dependencies:
```
pip install -r requirements.txt
pip install -i https://pypi.org/simple/ bitsandbytes
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
After the installation, you can run the app by executing:
```
python heb.py
```
This will start the Gradio interface locally, which you can access through the provided URL in your command line interface.

## How to Use

Once the application is running, follow these steps:

1. Enter your message in the input textbox.
2. Click "Send" to generate a response from the Hebrew-Mistral-7B model.
3. Adjust the generation parameters using the sliders and checkbox in the "Adjustments" section to customize the generated text.
4. The generated text will be displayed in real-time, and the conversation history will be maintained for context.

## Features

- Intuitive web-based interface built with Gradio.
- Adjustable generation parameters for customized output.
- Real-time display of generated text for an interactive experience.
- Conversation history for maintaining context across multiple interactions.
- Uses CUDA for accelerated processing if available.

---

<div align="center">

<h2>Hebrew Text Generation<br/><span style="font-size:12px">Powered by Hebrew-Mistral-7B</span></h2>

<div>
<a href='https://huggingface.co/yam-peleg/Hebrew-Mistral-7B' target='_blank'>Hebrew-Mistral-7B Model</a>&emsp;
</div>

<br>

## Acknowledgement

Special thanks to [Yam Peleg](https://huggingface.co/yam-peleg) for developing and sharing the Hebrew-Mistral-7B model, enabling the creation of powerful Hebrew language applications.

## Disclaimer

This project is intended for educational and development purposes. It leverages publicly available models and APIs. Please ensure to comply with the terms of use of the underlying models and frameworks.
