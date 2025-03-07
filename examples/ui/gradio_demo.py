import os
import asyncio
from dataclasses import dataclass
from typing import List, Optional
import json

# Third-party imports
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepseek
from pydantic import SecretStr
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import subprocess
import time
import requests

# Local module imports
from browser_use import Agent, Browser, BrowserConfig

load_dotenv()

# Model configurations
MODEL_CONFIGS = {
    'gpt-4': {
        'label': 'GPT-4 (OpenAI)',
        'placeholder': 'sk-...',
        'env_var': 'OPENAI_API_KEY'
    },
    'gpt-3.5-turbo': {
        'label': 'GPT-3.5 Turbo (OpenAI)',
        'placeholder': 'sk-...',
        'env_var': 'OPENAI_API_KEY'
    },
    'gemini-2.0-flash-exp': {
        'label': 'Gemini 2.0 Flash (Google)',
        'placeholder': 'Enter your Google API key',
        'env_var': 'GOOGLE_API_KEY'
    },
    'claude-3-opus': {
        'label': 'Claude 3 Opus (Anthropic)',
        'placeholder': 'Enter your Anthropic API key',
        'env_var': 'ANTHROPIC_API_KEY'
    },
    'deepseek-chat': {
        'label': 'Deepseek Chat',
        'placeholder': 'Enter your Deepseek API key',
        'env_var': 'DEEPSEEK_API_KEY'
    }
}

def load_config() -> Optional[str]:
    """Load API key from config file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'browser', 'config.json')
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            return config.get('gemini_api_key')
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None

def update_api_key_placeholder(model: str):
    """Updates the API key placeholder based on selected model"""
    return MODEL_CONFIGS[model]['placeholder']

def get_llm(model: str, api_key: str):
    """Creates an LLM instance based on the selected model"""
    if not api_key.strip():
        raise ValueError(f'Please provide an API key for {MODEL_CONFIGS[model]["label"]}')
    
    if model in ['gpt-4', 'gpt-3.5-turbo']:
        return ChatOpenAI(model=model, api_key=api_key)
    elif model == 'gemini-2.0-flash-exp':
        return ChatGoogleGenerativeAI(model=model, api_key=SecretStr(api_key))
    elif model == 'claude-3-opus':
        return ChatAnthropic(model=model, api_key=api_key)
    elif model == 'deepseek-chat':
        return ChatDeepseek(api_key=api_key)
    else:
        raise ValueError(f'Unsupported model: {model}')

@dataclass
class ActionResult:
	is_done: bool
	extracted_content: Optional[str]
	error: Optional[str]
	include_in_memory: bool


@dataclass
class AgentHistoryList:
	all_results: List[ActionResult]
	all_model_outputs: List[dict]


def parse_agent_history(history_str: str) -> None:
	console = Console()

	# Split the content into sections based on ActionResult entries
	sections = history_str.split('ActionResult(')

	for i, section in enumerate(sections[1:], 1):  # Skip first empty section
		# Extract relevant information
		content = ''
		if 'extracted_content=' in section:
			content = section.split('extracted_content=')[1].split(',')[0].strip("'")

		if content:
			header = Text(f'Step {i}', style='bold blue')
			panel = Panel(content, title=header, border_style='blue')
			console.print(panel)
			console.print()


# Kill any existing Chrome debugging sessions
def cleanup_chrome_instances():
    try:
        if os.name == 'posix':  # macOS or Linux
            subprocess.run(['pkill', '-f', 'Google Chrome.*--remote-debugging-port'], capture_output=True)
        elif os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/F', '/IM', 'chrome.exe'], capture_output=True)
    except Exception as e:
        print(f"Warning: Could not clean up Chrome instances: {e}")

async def run_browser_task(
    task: str,
    api_key: str,
    model: str,
    headless: bool = True,
) -> str:
    try:
        # Initialize the LLM based on selected model
        llm = get_llm(model, api_key)
        
        # Configure browser with Playwright's default browser
        browser = Browser(
            config=BrowserConfig(
                headless=headless,
                extra_chromium_args=[
                    '--no-first-run',
                    '--no-default-browser-check'
                ]
            )
        )
        
        agent = Agent(
            task=task,
            llm=llm,
            browser=browser
        )
        
        result = await agent.run()
        await browser.close()
        return result
    except Exception as e:
        return f'Error: {str(e)}'

def create_ui():
    with gr.Blocks(title='Browser Use GUI') as interface:
        gr.Markdown('# Browser Use Task Automation')
        gr.Markdown('''
        ### Important Notes:
        1. Make sure to provide the appropriate API key for your selected model
        2. The agent will access websites using Playwright's default browser
        3. Review tasks carefully before execution
        ''')

        with gr.Row():
            with gr.Column():
                model = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value='gemini-2.0-flash-exp',
                    label='Model'
                )
                api_key = gr.Textbox(
                    label='API Key',
                    placeholder=MODEL_CONFIGS['gemini-2.0-flash-exp']['placeholder'],
                    type='password'
                )
                task = gr.Textbox(
                    label='Task Description',
                    placeholder='E.g., Find flights from New York to London for next week',
                    lines=3,
                )
                headless = gr.Checkbox(label='Run Headless', value=False)
                submit_btn = gr.Button('Run Task')

            with gr.Column():
                output = gr.Textbox(label='Output', lines=10, interactive=False)

        # Update API key placeholder when model changes
        model.change(
            fn=update_api_key_placeholder,
            inputs=[model],
            outputs=[api_key]
        )

        submit_btn.click(
            fn=lambda *args: asyncio.run(run_browser_task(*args)),
            inputs=[task, api_key, model, headless],
            outputs=output,
        )

    return interface

if __name__ == '__main__':
    demo = create_ui()
    demo.launch()