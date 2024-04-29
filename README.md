<h1 align="center">Hebrew-Mistral-7B-GUI</h1>

<h3 align="center">GUI for Generating High-Quality Hebrew Text with Hebrew-Mistral-7B</h3>

<p align="left">
A user-friendly interface for interacting with the powerful Hebrew-Mistral-7B language model, allowing you to generate coherent and contextually relevant text in Hebrew with adjustable parameters for customized output.
</p>

![RTL](https://github.com/ShmuelRonen/Hebrew-Mistral-7B-GUI/assets/80190186/d2774e4f-ed20-4fb4-95dc-32000b217e3c)

#### Update: Three Additional Functions

The code includes three additional functions to enhance the user experience:

1. **Create Paragraphs**: When enabled, this function automatically divides the generated response into paragraphs for better readability. The number of sentences per paragraph can be adjusted in the code.

2. **Remove Paragraphs**: Clicking this button removes the paragraph formatting from the user's input text, converting it into a continuous block of text. This can be useful when providing a long input text without paragraph breaks.

3. **Copy Last Response**: Clicking this button copies the last generated response from the bot to the user's input textbox. This allows the user to easily continue the conversation based on the bot's previous response or use it as a starting point for a new query.

These functions provide additional convenience and flexibility for interacting with the Hebrew-Mistral-7B model through the GUI.

```
git pull
```


#### update: now the GUI write text in RTL mode:
```
git pull
```

## Installation

### One-click Installation

1. Clone the repository:
```
git clone https://github.com/your-username/Hebrew-Mistral-7B-GUI.git
cd Hebrew-Mistral-7B-GUI
```
```
init_env.bat
```
The script will automatically set up the virtual environment and install the required dependencies.

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
